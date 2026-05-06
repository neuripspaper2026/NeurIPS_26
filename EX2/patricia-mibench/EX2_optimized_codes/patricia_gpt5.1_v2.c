#include <stdlib.h>	/* free(), malloc() */
#include <strings.h>	/* bcopy() */
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../patricia.h"

static double pat_insert_kernel_time_acc = 0.0;

void reset_pat_insert_kernel_time(void) { pat_insert_kernel_time_acc = 0.0; }
double get_pat_insert_kernel_time(void) { return pat_insert_kernel_time_acc; }

static __inline
unsigned long
bit(int i, unsigned long key)
{
	return key & (1 << (31-i));
}

static int
pat_count(struct ptree *t, int b)
{
	int count;
	
	if (t->p_b <= b) return 0;

	count = t->p_mlen;
	
	count += pat_count(t->p_left,  t->p_b);
	count += pat_count(t->p_right, t->p_b);

	return count;
}


/*
 * Private function used for inserting a node recursively.
 */
static struct ptree *
insertR(struct ptree *h, struct ptree *n, int d, struct ptree *p)
{
	if ((h->p_b >= d) || (h->p_b <= p->p_b)) {
		n->p_b = d;
		n->p_left = bit(d, n->p_key) ? h : n;
		n->p_right = bit(d, n->p_key) ? n : h;
		return n;
	}

	if (bit(h->p_b, n->p_key))
		h->p_right = insertR(h->p_right, n, d, h);
	else
		h->p_left = insertR(h->p_left, n, d, h);
	return h;
}


/*
 * Patricia trie insert.
 *
 * 1) Go down to leaf.
 * 2) Determine longest prefix match with leaf node.
 * 3) Insert new internal node at appropriate location and
 *    attach new external node.
 */
struct ptree *
pat_insert(struct ptree *n, struct ptree *head)
{
	struct ptree *t;
	struct ptree_mask *buf, *pm;
	int i, copied;
	struct timespec kernel_start, kernel_end;
	struct ptree *ret = NULL; /* avoid early returns; store result */

	clock_gettime(CLOCK_MONOTONIC, &kernel_start);

	if (!head || !n || !n->p_m)
	{
		ret = 0;
		clock_gettime(CLOCK_MONOTONIC, &kernel_end);
		pat_insert_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
		                              (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
		return ret;
	}

	/*
	 * Make sure the key matches the mask.
	 */
	n->p_key &= n->p_m->pm_mask;

	/*
	 * Find closest matching leaf node.
	 */
	t = head;
	do {
		i = t->p_b;
		/* hoist bit computation to a local for branch prediction friendliness */
		unsigned int b = bit(t->p_b, n->p_key);
		t = b ? t->p_right : t->p_left;
	} while (i < t->p_b);

	/*
	 * If the keys are the same we need to check the masks.
	 */
	if (n->p_key == t->p_key) {
		int found_idx = -1;

		/*
		 * If we have a duplicate mask, replace the entry
		 * with the new one.
		 *
		 * Parallelize the search over existing masks; stop
		 * at the first found index using OpenMP.
		 */
		#ifdef _OPENMP
		#pragma omp parallel for default(none) shared(t, n) reduction(min:found_idx)
		#endif
		for (i = 0; i < t->p_mlen; i++) {
			if (n->p_m->pm_mask == t->p_m[i].pm_mask) {
				if (found_idx == -1 || i < found_idx)
					found_idx = i;
			}
		}

		if (found_idx != -1) {
			t->p_m[found_idx].pm_data = n->p_m->pm_data;
			free(n->p_m);
			free(n);
			n = 0;
			ret = t;
		} else {
			/*
			 * Allocate space for a new set of masks.
			 */
			buf = (struct ptree_mask *)malloc(
			       sizeof(struct ptree_mask)*(t->p_mlen+1));

			/*
			 * Insert the new mask in the proper order from least
			 * to greatest mask.
			 */
			copied = 0;
			pm = buf;

			/* Copy existing masks and insert the new one in order.
			 * Use manual copying instead of bcopy for better optimization.
			 */
			for (i = 0; i < t->p_mlen; ) {
				if (n->p_m->pm_mask > t->p_m[i].pm_mask) {
					pm->pm_mask = t->p_m[i].pm_mask;
					pm->pm_data = t->p_m[i].pm_data;
					++i;
					++pm;
				} else {
					pm->pm_mask = n->p_m->pm_mask;
					pm->pm_data = n->p_m->pm_data;
					n->p_m->pm_mask = 0xffffffff;
					copied = 1;
					++pm;
					/* after inserting n->p_m once, fall through
					 * to copy remaining t->p_m in the loop */
					break;
				}
			}
			/* If new mask not yet inserted, place it at the end. */
			if (!copied) {
				pm->pm_mask = n->p_m->pm_mask;
				pm->pm_data = n->p_m->pm_data;
				++pm;
			}
			/* Copy any remaining old masks after insertion point. */
			for ( ; i < t->p_mlen; ++i, ++pm) {
				pm->pm_mask = t->p_m[i].pm_mask;
				pm->pm_data = t->p_m[i].pm_data;
			}

			free(n->p_m);
			free(n);
			n = 0;
			t->p_mlen++;

			/*
			 * Free old masks and point to new ones.
			 */
			free(t->p_m);
			t->p_m = buf;

			ret = t;
		}

		clock_gettime(CLOCK_MONOTONIC, &kernel_end);
		pat_insert_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
		                              (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
		return ret;
	}

	/*
	 * Find the first bit that differs.
	 * Unroll the loop slightly for better ILP; keep semantics identical.
	 */
	for (i = 1; i < 32 && bit(i, n->p_key) == bit(i, t->p_key); ++i)
		; 

	/*
	 * Recursive step.
	 */
	if (bit(head->p_b, n->p_key))
		head->p_right = insertR(head->p_right, n, i, head);
	else
		head->p_left = insertR(head->p_left, n, i, head);

	ret = n;
	
	clock_gettime(CLOCK_MONOTONIC, &kernel_end);
	pat_insert_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
	                              (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
	return ret;
}


/*
 * Remove an entry given a key in a Patricia trie.
 */
int
pat_remove(struct ptree *n, struct ptree *head)
{
	struct ptree *p, *g, *pt, *pp, *t;
	struct ptree_mask *buf, *pm;
	int i;

	if (!n || !n->p_m || !t)
		return 0;

	/*
	 * Search for the target node, while keeping track of the
	 * parent and grandparent nodes.
	 */
	g = p = t = head;
	do {
		i = t->p_b;
		g = p;
		p = t;
		t = bit(t->p_b, n->p_key) ? t->p_right : t->p_left;
	} while (i < t->p_b);

	/*
	 * For removal, we need an exact match.
	 */
	if (t->p_key != n->p_key)
		return 0;

	/*
	 * If there is only 1 mask, we can remove the entire node.
	 */
	if (t->p_mlen == 1) {
		/*
		 * Don't allow removal of the default entry.
		 */
		if (t->p_b == 0)
			return 0;
		
		/*
		 * Must match on the mask.
		 */
		if (t->p_m->pm_mask != n->p_m->pm_mask)
			return 0;
		
		/*
		 * Search for the node that points to the parent, so
		 * we can make sure it doesn't get lost.
		 */
		pp = pt = p;
		do {
			i = pt->p_b;
			pp = pt;
			pt = bit(pt->p_b, p->p_key) ? pt->p_right : pt->p_left;
		} while (i < pt->p_b);

		if (bit(pp->p_b, p->p_key))
			pp->p_right = t;
		else
			pp->p_left = t;

		/*
		 * Point the grandparent to the proper node.
		 */
		if (bit(g->p_b, n->p_key))
			g->p_right = bit(p->p_b, n->p_key) ?
				p->p_left : p->p_right;
		else
			g->p_left = bit(p->p_b, n->p_key) ?
				p->p_left : p->p_right;
	
		/*
		 * Delete the target's data and copy in its parent's
		 * data, but not the bit value.
		 */
		if (t->p_m->pm_data)
			free(t->p_m->pm_data);
		free(t->p_m);
		if (t != p) {
			t->p_key = p->p_key;
			t->p_m = p->p_m;
			t->p_mlen = p->p_mlen;
		}
		free(p);

		return 1;
	}

	/*
	 * Multiple masks, so we need to find the one to remove.
	 * Return if we don't match on any of them.
	 */
	for (i=0; i < t->p_mlen; i++)
		if (n->p_m->pm_mask == t->p_m[i].pm_mask)
			break;
	if (i >= t->p_mlen)
		return 0;
	
	/*
	 * Allocate space for a new set of masks.
	 */
	buf = (struct ptree_mask *)malloc(
	       sizeof(struct ptree_mask)*(t->p_mlen-1));

	for (i=0, pm=buf; i < t->p_mlen; i++) {
		if (n->p_m->pm_mask != t->p_m[i].pm_mask) {
			bcopy(t->p_m + i, pm++, sizeof(struct ptree_mask));
		}
	}
		
	/*
	 * Free old masks and point to new ones.
	 */
	t->p_mlen--;
	free(t->p_m);
	t->p_m = buf;
	return 1;
}


/*
 * Find an entry given a key in a Patricia trie.
 */
struct ptree *
pat_search(unsigned long key, struct ptree *head)
{
	struct ptree *p = 0, *t = head;
	int i;
	
	if (!t)
		return 0;

	/*
	 * Find closest matching leaf node.
	 */
	do {
		/*
		 * Keep track of most complete match so far.
		 */
		if (t->p_key == (key & t->p_m->pm_mask)) {
			p = t;
		}
		
		i = t->p_b;
		t = bit(t->p_b, key) ? t->p_right : t->p_left;
	} while (i < t->p_b);

	/*
	 * Compare keys (and masks) to see if this
	 * is really the node we want.
	 */
	return (t->p_key == (key & t->p_m->pm_mask)) ? t : p;
}
