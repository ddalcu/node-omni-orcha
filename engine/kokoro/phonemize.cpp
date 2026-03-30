#include "phonemize.h"
#include "vocab.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace kokoro {

// ─── Dictionary ───

struct DictEntry {
    std::vector<int16_t> tokens;
};

static std::unordered_map<std::string, DictEntry> g_dict;
static bool g_loaded = false;
static std::mutex g_mutex;

bool g2p_init(const std::string& dict_path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_loaded) return true;

    std::ifstream f(dict_path, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "[kokoro] Cannot open phoneme dict: %s\n", dict_path.c_str());
        return false;
    }

    // Read header
    char magic[4];
    f.read(magic, 4);
    if (!f || memcmp(magic, "PHON", 4) != 0) {
        fprintf(stderr, "[kokoro] Invalid phoneme dict magic\n");
        return false;
    }

    uint16_t version = 0;
    f.read(reinterpret_cast<char*>(&version), 2);
    if (version != 1) {
        fprintf(stderr, "[kokoro] Unsupported phoneme dict version: %d\n", version);
        return false;
    }

    uint32_t num_entries = 0;
    f.read(reinterpret_cast<char*>(&num_entries), 4);
    if (!f || num_entries == 0 || num_entries > 500000) {
        fprintf(stderr, "[kokoro] Invalid entry count: %u\n", num_entries);
        return false;
    }

    g_dict.reserve(num_entries);

    for (uint32_t i = 0; i < num_entries; i++) {
        uint8_t word_len = 0;
        f.read(reinterpret_cast<char*>(&word_len), 1);
        if (!f || word_len == 0) break;

        std::string word(word_len, '\0');
        f.read(&word[0], word_len);

        uint8_t num_tokens = 0;
        f.read(reinterpret_cast<char*>(&num_tokens), 1);

        DictEntry entry;
        entry.tokens.resize(num_tokens);
        f.read(reinterpret_cast<char*>(entry.tokens.data()), num_tokens * 2);

        if (!f) break;
        g_dict[word] = std::move(entry);
    }

    g_loaded = true;
    fprintf(stderr, "[kokoro] Loaded phoneme dict: %zu words\n", g_dict.size());
    return true;
}

void g2p_cleanup() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_dict.clear();
    g_loaded = false;
}

bool g2p_ready() {
    return g_loaded;
}

// ─── Text normalization ───

static const std::unordered_map<std::string, std::string>& abbreviations() {
    static const std::unordered_map<std::string, std::string> abbr = {
        {"mr", "mister"}, {"mrs", "missus"}, {"ms", "miss"}, {"dr", "doctor"},
        {"prof", "professor"}, {"sr", "senior"}, {"jr", "junior"},
        {"st", "saint"}, {"ave", "avenue"}, {"blvd", "boulevard"},
        {"govt", "government"}, {"dept", "department"}, {"etc", "etcetera"},
        {"vs", "versus"}, {"jan", "january"}, {"feb", "february"},
        {"mar", "march"}, {"apr", "april"}, {"jun", "june"}, {"jul", "july"},
        {"aug", "august"}, {"sep", "september"}, {"oct", "october"},
        {"nov", "november"}, {"dec", "december"},
    };
    return abbr;
}

// Function word reductions — CMU dict has citation (strong) forms but in
// connected speech these words use weak/unstressed vowels. These IPA strings
// match espeak-ng's output for American English (what Kokoro was trained on).
static const std::unordered_map<std::string, std::string> g_weak_forms = {
    {"to",     "t\u0259"},              // tə  (not tuː)
    {"the",    "\u00F0\u0259"},          // ðə  (not ðiː)
    {"a",      "\u0259"},               // ə   (not eɪ)
    {"an",     "\u0259n"},              // ən
    {"of",     "\u0259v"},              // əv  (not ɑːv)
    {"for",    "f\u0259\u0279"},        // fəɹ (not fɔːɹ)
    {"and",    "\u0259nd"},             // ənd (not ænd)
    {"or",     "\u0259\u0279"},          // əɹ
    {"but",    "b\u0259t"},             // bət
    {"at",     "\u0259t"},              // ət
    {"from",   "f\u0279\u0259m"},       // fɹəm
    {"as",     "\u0259z"},              // əz
    {"was",    "w\u0259z"},             // wəz
    {"were",   "w\u0259\u0279"},        // wəɹ
    {"are",    "\u0259\u0279"},          // əɹ
    {"can",    "k\u0259n"},             // kən
    {"had",    "h\u0259d"},             // həd
    {"has",    "h\u0259z"},             // həz
    {"have",   "h\u0259v"},             // həv
    {"been",   "b\u026An"},             // bɪn
    {"some",   "s\u0259m"},             // səm
    {"than",   "\u00F0\u0259n"},         // ðən
    {"that",   "\u00F0\u0259t"},         // ðət
    {"them",   "\u00F0\u0259m"},         // ðəm
    {"will",   "w\u0259l"},             // wəl
    {"would",  "w\u0259d"},             // wəd
    {"could",  "k\u0259d"},             // kəd
    {"should", "\u0283\u0259d"},         // ʃəd
    {"shall",  "\u0283\u0259l"},         // ʃəl
    {"must",   "m\u0259st"},            // məst
    {"just",   "\u02A4\u0259st"},        // ʤəst
    {"with",   "w\u026A\u00F0"},        // wɪð
    {"your",   "j\u0259\u0279"},        // jəɹ
    {"his",    "\u026Az"},              // ɪz
    {"her",    "h\u0259\u0279"},        // həɹ
};

static const char* ones[] = {
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen"
};

static const char* tens[] = {
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
};

static std::string number_to_words(int64_t n) {
    if (n < 0) return "minus " + number_to_words(-n);
    if (n == 0) return "zero";

    std::string result;

    if (n >= 1000000000) {
        result += number_to_words(n / 1000000000) + " billion ";
        n %= 1000000000;
    }
    if (n >= 1000000) {
        result += number_to_words(n / 1000000) + " million ";
        n %= 1000000;
    }
    if (n >= 1000) {
        result += number_to_words(n / 1000) + " thousand ";
        n %= 1000;
    }
    if (n >= 100) {
        result += std::string(ones[n / 100]) + " hundred ";
        n %= 100;
    }
    if (n >= 20) {
        result += std::string(tens[n / 10]) + " ";
        n %= 10;
    }
    if (n > 0) {
        result += std::string(ones[n]) + " ";
    }

    // Trim trailing space
    while (!result.empty() && result.back() == ' ') result.pop_back();
    return result;
}

// Expand numbers in text to words
static std::string expand_numbers(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        if (std::isdigit(static_cast<unsigned char>(text[i]))) {
            // Collect digits
            size_t start = i;
            while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) i++;
            std::string num_str = text.substr(start, i - start);
            // Handle very large numbers by reading them digit-by-digit
            if (num_str.size() > 12) {
                for (char c : num_str) {
                    if (!result.empty()) result += ' ';
                    result += ones[c - '0'];
                }
            } else {
                int64_t n = std::stoll(num_str);
                if (!result.empty() && result.back() != ' ') result += ' ';
                result += number_to_words(n);
            }
        } else {
            result += text[i++];
        }
    }
    return result;
}

// ─── Letter-to-sound fallback for unknown words ───

// Simple English letter-to-phoneme rules. Not perfect, but handles
// common patterns for names and neologisms.
static std::vector<int16_t> letter_to_sound(const std::string& word) {
    const auto& map = symbol_to_id();
    std::vector<int16_t> tokens;

    // Simple mapping: each letter → most common phoneme
    // This is intentionally basic — the dictionary handles 99%+ of real words
    static const std::unordered_map<char, std::vector<uint32_t>> letter_map = {
        {'a', {0x00E6}},       // æ
        {'b', {'b'}},
        {'c', {'k'}},
        {'d', {'d'}},
        {'e', {0x025B}},       // ɛ
        {'f', {'f'}},
        {'g', {0x0261}},       // ɡ
        {'h', {'h'}},
        {'i', {0x026A}},       // ɪ
        {'j', {0x02A4}},       // ʤ
        {'k', {'k'}},
        {'l', {'l'}},
        {'m', {'m'}},
        {'n', {'n'}},
        {'o', {0x0251}},       // ɑ
        {'p', {'p'}},
        {'q', {'k'}},
        {'r', {0x0279}},       // ɹ
        {'s', {'s'}},
        {'t', {'t'}},
        {'u', {0x028C}},       // ʌ
        {'v', {'v'}},
        {'w', {'w'}},
        {'x', {'k', 's'}},
        {'y', {'j'}},
        {'z', {'z'}},
    };

    for (size_t i = 0; i < word.size(); i++) {
        char c = word[i];

        // Common digraphs
        if (i + 1 < word.size()) {
            std::string di = word.substr(i, 2);
            if (di == "th") { auto it = map.find(0x03B8); if (it != map.end()) tokens.push_back(it->second); i++; continue; }
            if (di == "sh") { auto it = map.find(0x0283); if (it != map.end()) tokens.push_back(it->second); i++; continue; }
            if (di == "ch") { auto it = map.find(0x02A7); if (it != map.end()) tokens.push_back(it->second); i++; continue; }
            if (di == "ph") { tokens.push_back(map.at('f')); i++; continue; }
            if (di == "ng") { auto it = map.find(0x014B); if (it != map.end()) tokens.push_back(it->second); i++; continue; }
            if (di == "ck") { tokens.push_back(map.at('k')); i++; continue; }
            if (di == "wh") { tokens.push_back(map.at('w')); i++; continue; }
        }

        auto lm_it = letter_map.find(c);
        if (lm_it != letter_map.end()) {
            for (uint32_t cp : lm_it->second) {
                auto it = map.find(cp);
                if (it != map.end()) tokens.push_back(it->second);
            }
        }
    }

    return tokens;
}

// ─── Tokenize text ───

static std::string to_lower(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) out += std::tolower(static_cast<unsigned char>(c));
    return out;
}

// Split text into words and punctuation tokens
static std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;

    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];

        if (std::isalpha(static_cast<unsigned char>(c)) || c == '\'') {
            current += c;
        } else {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                tokens.push_back(" ");
            } else if (c == '.' || c == ',' || c == '!' || c == '?' ||
                       c == ';' || c == ':') {
                tokens.push_back(std::string(1, c));
            }
            // Other characters (brackets, etc.) silently skipped
        }
    }
    if (!current.empty()) tokens.push_back(current);

    return tokens;
}

// ─── Phonological post-processing (American English allophonic rules) ───
// These context-dependent rules match what espeak-ng produces and Kokoro was
// trained on, but a dictionary can't capture since they depend on neighbors.

// Token IDs (must match vocab.h)
static constexpr int64_t TK_PAD   = 0;
static constexpr int64_t TK_SPACE = 16;
static constexpr int64_t TK_t     = 62;   // /t/
static constexpr int64_t TK_d     = 46;   // /d/
static constexpr int64_t TK_l     = 54;   // /l/ (light)
static constexpr int64_t TK_FLAP  = 125;  // ɾ (alveolar flap)
static constexpr int64_t TK_DARK_L= 106;  // ɫ (velarized L)
static constexpr int64_t TK_STRESS1 = 156; // ˈ
static constexpr int64_t TK_STRESS2 = 157; // ˌ
static constexpr int64_t TK_LONG  = 158;  // ː

static bool is_vowel_token(int64_t t) {
    // ASCII vowels used in IPA diphthongs: a(43) e(47) i(51) o(57) u(63)
    // IPA vowels: ɑ(69) ɐ(70) ɒ(71) æ(72) ɔ(76) ə(83) ɘ(84) ɚ(85) ɛ(86)
    //   ɜ(87) ɝ(88) ɞ(89) ɨ(101) ɪ(102) ʉ(134) ʊ(135) ʌ(138) ɤ(140) ʏ(144)
    switch (t) {
        case 43: case 47: case 51: case 57: case 63:       // a e i o u
        case 69: case 70: case 71: case 72: case 76:       // ɑ ɐ ɒ æ ɔ
        case 83: case 84: case 85: case 86: case 87: case 88: case 89: // ə ɘ ɚ ɛ ɜ ɝ ɞ
        case 101: case 102: case 134: case 135: case 138: case 140: case 144:
            return true;
        default:
            return false;
    }
}

static bool is_modifier_token(int64_t t) {
    return t == TK_STRESS1 || t == TK_STRESS2 || t == TK_LONG;
}

// Look past modifiers (stress/length markers) to find the real phoneme
static int64_t peek_skip_modifiers(const std::vector<int64_t>& ids, size_t pos, int dir) {
    for (size_t i = pos + dir; i > 0 && i < ids.size() - 1; i += dir) {
        if (!is_modifier_token(ids[i])) return ids[i];
    }
    return TK_PAD;
}

static void apply_phonological_rules(std::vector<int64_t>& ids) {
    if (ids.size() < 3) return;

    for (size_t i = 1; i + 1 < ids.size(); i++) {
        // --- Intervocalic t/d flapping ---
        // American English: /t/ or /d/ between vowels → ɾ (flap)
        // ONLY when NOT part of a stressed syllable onset.
        // The stress marker ˈ/ˌ is placed BEFORE the onset consonant in IPA,
        // so for "hotel" = [ʊ, ˈ, t, ɛ] the ˈ is BEHIND the t.
        // e.g. "better"→bɛɾəɹ, "water"→wɔːɾəɹ BUT NOT "hotel"→hoʊˈtɛl
        if (ids[i] == TK_t || ids[i] == TK_d) {
            int64_t prev = peek_skip_modifiers(ids, i, -1);
            int64_t next = peek_skip_modifiers(ids, i, +1);
            // Check backward: is there a stress marker between the prev vowel and this t/d?
            bool stressed_onset = false;
            for (size_t j = i; j > 0; j--) {
                if (ids[j] == TK_STRESS1 || ids[j] == TK_STRESS2) { stressed_onset = true; break; }
                if (is_vowel_token(ids[j]) || ids[j] == TK_SPACE || ids[j] == TK_PAD) break;
            }
            if (is_vowel_token(prev) && is_vowel_token(next) && !stressed_onset) {
                ids[i] = TK_FLAP;
            }
        }

        // Note: dark L (l→ɫ) rule removed — espeak-ng uses plain l for
        // American English, which is what Kokoro was trained on.
    }
}

// ─── Public API ───

std::vector<int64_t> text_to_token_ids(const std::string& text) {
    if (!g_loaded) return {};

    const auto& sym_map = symbol_to_id();

    // Normalize text
    std::string normalized = expand_numbers(text);

    // Tokenize
    auto tokens = tokenize(normalized);

    // Convert each token to phoneme IDs
    std::vector<int64_t> ids;
    ids.push_back(0); // leading pad

    for (const auto& tok : tokens) {
        if (tok == " ") {
            // Word boundary — add space token
            auto it = sym_map.find(' ');
            if (it != sym_map.end()) ids.push_back(it->second);
            continue;
        }

        // Check if it's punctuation
        if (tok.size() == 1) {
            char c = tok[0];
            if (c == '.' || c == ',' || c == '!' || c == '?' || c == ';' || c == ':') {
                auto it = sym_map.find(static_cast<uint32_t>(c));
                if (it != sym_map.end()) ids.push_back(it->second);
                continue;
            }
        }

        // Word — look up in dictionary
        std::string word = to_lower(tok);

        // Strip trailing punctuation from word
        while (!word.empty() && !std::isalpha(static_cast<unsigned char>(word.back())) && word.back() != '\'') {
            word.pop_back();
        }
        if (word.empty()) continue;

        // Function word reductions — weak/unstressed forms for natural speech
        auto weak_it = g_weak_forms.find(word);
        if (weak_it != g_weak_forms.end()) {
            auto reduced = phonemes_to_ids(weak_it->second);
            for (size_t i = 1; i + 1 < reduced.size(); i++) {
                ids.push_back(reduced[i]);
            }
            continue;
        }

        // Check abbreviations
        auto& abbr = abbreviations();
        auto abbr_it = abbr.find(word);
        if (abbr_it != abbr.end()) {
            auto expanded = text_to_token_ids(abbr_it->second);
            for (size_t i = 1; i + 1 < expanded.size(); i++) {
                ids.push_back(expanded[i]);
            }
            continue;
        }

        // Dictionary lookup
        auto dict_it = g_dict.find(word);
        if (dict_it != g_dict.end()) {
            for (int16_t t : dict_it->second.tokens) {
                ids.push_back(static_cast<int64_t>(t));
            }
        } else {
            // Fallback: letter-to-sound rules
            auto fallback = letter_to_sound(word);
            for (int16_t t : fallback) {
                ids.push_back(static_cast<int64_t>(t));
            }
        }
    }

    ids.push_back(0); // trailing pad

    // --- Post-processing: American English phonological rules ---
    // These apply context-dependent allophonic changes that espeak-ng produces
    // and Kokoro was trained on, but a dictionary can't capture.
    apply_phonological_rules(ids);

    return ids;
}

} // namespace kokoro
