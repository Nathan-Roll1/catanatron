CC = cc
CFLAGS = -O3 -march=native -flto -Wall -Wextra -Wno-unused-parameter
LDFLAGS = -flto -lm

SRCS = rng.c map.c board.c state.c actions.c apply_action.c game.c value.c search.c
OBJS = $(SRCS:.c=.o)

all: catanatron

catanatron: main.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_rng: test_rng.c rng.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_map: test_map.c map.o rng.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_board: test_board.c board.o map.o rng.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_state: test_state.c state.o board.o map.o rng.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_game: test_game.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o catanatron test_rng test_map test_board test_state test_game

.PHONY: all clean
