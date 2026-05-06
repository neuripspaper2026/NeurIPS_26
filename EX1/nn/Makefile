CC=gcc
CFLAGS=-O3
LDFLAGS=-lm

SRCS=nn.c
OBJS=$(SRCS:.c=.o)
EXE=nn

all: $(EXE)

$(EXE): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(EXE) $(OBJS) output.txt