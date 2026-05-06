CC      = gcc
CFLAGS  = -O3 -std=gnu89 -w
TARGET  = cholesky

SRCS = amal.C assign.C bfac.C bksolve.C block2.C fo.C mf.C numLL.C        parts.C seg.C solve.C tree.C util.C malloc.C
OBJS = $(SRCS:.C=.o)

.SUFFIXES:

all: $(TARGET)

# Compile .C files directly as C, and include null_macros.h to replace m4
%.o: %.C
	$(CC) -x c $(CFLAGS) -I. -include null_macros.h -c -o $@ $<

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) -lm

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean
