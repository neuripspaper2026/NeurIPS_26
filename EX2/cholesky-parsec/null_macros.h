#ifndef NULL_MACROS_H
#define NULL_MACROS_H

#include <time.h>
#include <stdlib.h>

/* Environment / init macros (no-op in single-thread build) */
#define EXTERN_ENV
#define MAIN_ENV
#define MAIN_INITENV(...)
/* Some code expects MAIN_END token */
#define MAIN_END

/* Locks / barriers become no-ops */
#define LOCKDEC(x)
#define LOCKINIT(x)
#define LOCK(x)
#define UNLOCK(x)
#define BARDEC(x)
#define BARINIT(x)
#define BAR(x, n)

/* Timing helper (tolerates missing semicolon at call site) */
#define CLOCK(x) (x) = clock();

/* Global allocator: map to system malloc in single-thread build */
#define G_MALLOC(sz, home) malloc((sz))

#endif /* NULL_MACROS_H */


