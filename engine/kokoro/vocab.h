#pragma once
// Kokoro-82M phoneme vocabulary.
// Maps Unicode code points to token IDs matching the Python reference:
//   _pad + _punctuation + _letters + _letters_ipa

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace kokoro {

// Build the symbol-to-ID map on first use.
inline const std::unordered_map<uint32_t, int>& symbol_to_id() {
    static const auto map = []() {
        // Exact symbol list from Kokoro's Python vocabulary.
        // Each entry is a single Unicode code point; the index is the token ID.
        // Order: pad($), punctuation, A-Z, a-z, IPA symbols.
        static const uint32_t symbols[] = {
            // 0: pad
            '$',
            // 1-16: punctuation + space
            ';', ':', ',', '.', '!', '?',
            0x00A1, // ¡
            0x00BF, // ¿
            0x2014, // —
            0x2026, // …
            0x201C, // "
            0x00AB, // «
            0x00BB, // »
            0x201D, // "
            '"',    // straight quote
            ' ',
            // 17-42: A-Z
            'A','B','C','D','E','F','G','H','I','J','K','L','M',
            'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
            // 43-68: a-z
            'a','b','c','d','e','f','g','h','i','j','k','l','m',
            'n','o','p','q','r','s','t','u','v','w','x','y','z',
            // 69+: IPA symbols
            0x0251, // ɑ
            0x0250, // ɐ
            0x0252, // ɒ
            0x00E6, // æ
            0x0253, // ɓ
            0x0299, // ʙ
            0x03B2, // β
            0x0254, // ɔ
            0x0255, // ɕ
            0x00E7, // ç
            0x0257, // ɗ
            0x0256, // ɖ
            0x00F0, // ð
            0x02A4, // ʤ
            0x0259, // ə
            0x0258, // ɘ
            0x025A, // ɚ
            0x025B, // ɛ
            0x025C, // ɜ
            0x025D, // ɝ
            0x025E, // ɞ
            0x025F, // ɟ
            0x0284, // ʄ
            0x0261, // ɡ
            0x0260, // ɠ
            0x0262, // ɢ
            0x029B, // ʛ
            0x0266, // ɦ
            0x0267, // ɧ
            0x0127, // ħ
            0x0265, // ɥ
            0x029C, // ʜ
            0x0268, // ɨ
            0x026A, // ɪ
            0x029D, // ʝ
            0x026D, // ɭ
            0x026C, // ɬ
            0x026B, // ɫ
            0x026E, // ɮ
            0x029F, // ʟ
            0x0271, // ɱ
            0x026F, // ɯ
            0x0270, // ɰ
            0x014B, // ŋ
            0x0273, // ɳ
            0x0272, // ɲ
            0x0274, // ɴ
            0x00F8, // ø
            0x0275, // ɵ
            0x0278, // ɸ
            0x03B8, // θ
            0x0153, // œ
            0x0276, // ɶ
            0x0298, // ʘ
            0x0279, // ɹ
            0x027A, // ɺ
            0x027E, // ɾ
            0x027B, // ɻ
            0x0280, // ʀ
            0x0281, // ʁ
            0x027D, // ɽ
            0x0282, // ʂ
            0x0283, // ʃ
            0x0288, // ʈ
            0x02A7, // ʧ
            0x0289, // ʉ
            0x028A, // ʊ
            0x028B, // ʋ
            0x2C71, // ⱱ
            0x028C, // ʌ
            0x0263, // ɣ
            0x0264, // ɤ
            0x028D, // ʍ
            0x03C7, // χ
            0x028E, // ʎ
            0x028F, // ʏ
            0x0291, // ʑ
            0x0290, // ʐ
            0x0292, // ʒ
            0x0294, // ʔ
            0x02A1, // ʡ
            0x0295, // ʕ
            0x02A2, // ʢ
            0x01C0, // ǀ
            0x01C1, // ǁ
            0x01C2, // ǂ
            0x01C3, // ǃ
            0x02C8, // ˈ
            0x02CC, // ˌ
            0x02D0, // ː
            0x02D1, // ˑ
            0x02BC, // ʼ
            0x02B4, // ʴ
            0x02B0, // ʰ
            0x02B1, // ʱ
            0x02B2, // ʲ
            0x02B7, // ʷ
            0x02E0, // ˠ
            0x02E4, // ˤ
            0x02DE, // ˞
            0x2193, // ↓
            0x2191, // ↑
            0x2192, // →
            0x2197, // ↗
            0x2198, // ↘
            0x0027, // '
            0x0303, // ̃ (combining tilde)
            0x0308, // ̈ (combining diaeresis)
            0x0304, // ̄ (combining macron)
            0x0300, // ̀ (combining grave)
            0x030A, // ̊ (combining ring above)
            0x1D7B, // ᵻ
        };

        constexpr int n = sizeof(symbols) / sizeof(symbols[0]);
        std::unordered_map<uint32_t, int> m;
        m.reserve(n);
        for (int i = 0; i < n; i++) {
            m[symbols[i]] = i;
        }
        return m;
    }();
    return map;
}

// Decode a UTF-8 string into a sequence of Unicode code points.
inline std::vector<uint32_t> utf8_to_codepoints(const std::string& s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    const uint8_t* p = reinterpret_cast<const uint8_t*>(s.data());
    const uint8_t* end = p + s.size();
    while (p < end) {
        uint32_t cp;
        if ((*p & 0x80) == 0) {
            cp = *p++;
        } else if ((*p & 0xE0) == 0xC0) {
            cp = (*p++ & 0x1F) << 6;
            if (p < end) cp |= (*p++ & 0x3F);
        } else if ((*p & 0xF0) == 0xE0) {
            cp = (*p++ & 0x0F) << 12;
            if (p < end) cp |= (*p++ & 0x3F) << 6;
            if (p < end) cp |= (*p++ & 0x3F);
        } else if ((*p & 0xF8) == 0xF0) {
            cp = (*p++ & 0x07) << 18;
            if (p < end) cp |= (*p++ & 0x3F) << 12;
            if (p < end) cp |= (*p++ & 0x3F) << 6;
            if (p < end) cp |= (*p++ & 0x3F);
        } else {
            p++; // skip invalid byte
            continue;
        }
        out.push_back(cp);
    }
    return out;
}

// Convert a phoneme string to Kokoro token IDs.
// Unknown symbols are silently skipped.
inline std::vector<int64_t> phonemes_to_ids(const std::string& phonemes) {
    const auto& map = symbol_to_id();
    auto cps = utf8_to_codepoints(phonemes);

    std::vector<int64_t> ids;
    ids.reserve(cps.size() + 2);

    // Add leading pad
    ids.push_back(0);

    for (auto cp : cps) {
        auto it = map.find(cp);
        if (it != map.end()) {
            ids.push_back(it->second);
        }
        // Unknown symbols silently skipped
    }

    // Add trailing pad
    ids.push_back(0);

    return ids;
}

} // namespace kokoro
