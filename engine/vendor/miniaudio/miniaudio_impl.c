#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_DEVICE_IO   /* decode only — no playback/capture */
#define MA_NO_ENCODING    /* we have our own WAV writer */
#include "miniaudio.h"
