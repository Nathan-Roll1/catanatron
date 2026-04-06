CC = cc
CFLAGS = -O3 -march=native -flto -Wall -Wextra -Wno-unused-parameter
LDFLAGS = -flto -lm
SRCDIR = src

SRCS = $(SRCDIR)/rng.c $(SRCDIR)/map.c $(SRCDIR)/board.c $(SRCDIR)/state.c \
       $(SRCDIR)/actions.c $(SRCDIR)/apply_action.c $(SRCDIR)/game.c \
       $(SRCDIR)/value.c $(SRCDIR)/search.c
OBJS = $(SRCS:.c=.o)

all: catanatron

catanatron: $(SRCDIR)/main.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(SRCDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -I$(SRCDIR) -c -o $@ $<

clean:
	rm -f $(SRCDIR)/*.o catanatron

.PHONY: all clean
