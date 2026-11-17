#ifndef MCKRILL_AVBD_LIST_PARAMS_INCLUDED
#define MCKRILL_AVBD_LIST_PARAMS_INCLUDED

// Constants of stage G (per-body constraint lists). Mirrored by Runtime/Constraints/AvbdConstraintLists.cs; a test checks the two against each other.

// Words of the control buffer.
#define AVBD_LIST_CONTROL_TOTAL   0 // entries in the list: the sum of the per-body counts. Written by the scan
#define AVBD_LIST_CONTROL_BROKEN  1 // joints and springs left out because they are fractured
#define AVBD_LIST_CONTROL_INVALID 2 // records or manifolds left out because they name a body slot that is not in use (or too many contacts)
#define AVBD_LIST_CONTROL_WORDS   4

// Threads of a scan group: a group scans the counts of AVBD_LIST_SCAN_GROUP bodies, and one group scans the sums of the groups, so the scan
// handles at most AVBD_LIST_SCAN_GROUP * AVBD_LIST_SCAN_GROUP bodies.
#define AVBD_LIST_SCAN_GROUP 1024

// Threads that sort the list of one body (a power of two that divides AVBD_GROUP_SIZE).
#define AVBD_LIST_SORT_LANES 8u

// Contact points per manifold slot (mirrors ContactLayout.MaxContactsPerPair): a manifold that reports more is invalid.
#define AVBD_LIST_CONTACTS_PER_SLOT 8

// A list entry is a constraint id (a joint or a spring), or, with this bit set, a manifold slot: one entry per manifold, whatever its number of
// contact points. The stages that walk a list read the pair record of the slot and loop over its points.
#define AVBD_LIST_CONTACT_FLAG 0x80000000u

#endif // MCKRILL_AVBD_LIST_PARAMS_INCLUDED
