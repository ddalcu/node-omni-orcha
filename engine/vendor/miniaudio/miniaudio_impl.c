#define STB_VORBIS_HEADER_ONLY
#include "../stb_vorbis.c"  /* OGG Vorbis support for miniaudio */

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_DEVICE_IO   /* decode only — no playback/capture */
#define MA_NO_ENCODING    /* we have our own WAV writer */
#include "miniaudio.h"

/* stb_vorbis implementation (after miniaudio) */
#undef STB_VORBIS_HEADER_ONLY
#include "../stb_vorbis.c"
