#ifndef VALUE_H
#define VALUE_H

#include "game.h"

double base_value_fn(Game *g, Color p0_color);
double base_value_fn_enemy_full(Game *g, Color p0_color);
double base_value_fn_known_future(Game *g, Color p0_color);
double base_value_fn_known_future_exact(Game *g, Color p0_color, bool use_exact_roll);
double base_value_fn_known_future_profile(Game *g, Color p0_color,
                                          bool use_exact_roll,
                                          int profile);

#endif
