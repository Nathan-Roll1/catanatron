#ifndef STATE_HASH_H
#define STATE_HASH_H

#include <stdint.h>

#include "game.h"

#ifdef __cplusplus
extern "C" {
#endif

uint64_t game_full_hash(const Game *g);
uint64_t game_dynamic_hash(const Game *g);

#ifdef __cplusplus
}
#endif

#endif
