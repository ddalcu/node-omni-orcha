#pragma once
// English grapheme-to-phoneme for Kokoro TTS.
// Uses a CMU Pronouncing Dictionary (binary format) with letter-to-sound
// fallback rules for unknown words. No external dependencies.

#include <cstdint>
#include <string>
#include <vector>

namespace kokoro {

// Load the phoneme dictionary from a binary file (phoneme_dict.bin).
// Returns true on success. Must be called before text_to_token_ids().
bool g2p_init(const std::string& dict_path);

// Free dictionary memory.
void g2p_cleanup();

// Whether the dictionary is loaded.
bool g2p_ready();

// Full pipeline: English text → Kokoro token IDs (with pad tokens).
std::vector<int64_t> text_to_token_ids(const std::string& text);

} // namespace kokoro
