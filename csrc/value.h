#ifndef VALUE_H
#define VALUE_H

#include "game.h"

double base_value_fn(Game *g, Color p0_color);
double base_value_fn_enemy_vp(Game *g, Color p0_color);
double base_value_fn_enemy_all_vp_prod(Game *g, Color p0_color);
double base_value_fn_enemy_leader(Game *g, Color p0_color);
double base_value_fn_enemy_full(Game *g, Color p0_color);

/* Runtime tuning for leaf_mode==4 (and direct enemy_full callers). */
void value_set_pressure_weight(double w);
void value_set_threat_bonus(double bonus, int vp_threshold);

#endif
