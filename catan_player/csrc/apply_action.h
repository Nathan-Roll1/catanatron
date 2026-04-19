#ifndef APPLY_ACTION_H
#define APPLY_ACTION_H

#include "state.h"

void apply_action(State *s, Action action, RngState *rng);
void apply_roll_forced(State *s, Action action, int dice_sum);
void apply_buy_dev_card_forced(State *s, Action action, int dev_card);
void apply_move_robber_forced(State *s, Action action, int forced_steal);

#endif
