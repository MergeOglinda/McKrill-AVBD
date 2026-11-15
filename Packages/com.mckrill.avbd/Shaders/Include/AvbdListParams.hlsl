#ifndef MCKRILL_AVBD_LIST_PARAMS_INCLUDED
#define MCKRILL_AVBD_LIST_PARAMS_INCLUDED

// Constants of stage G (per-body constraint lists). Mirrored by Runtime/Constraints/AvbdConstraintLists.cs; a test checks the two against each other.

// Words of the control buffer.
#define AVBD_LIST_CONTROL_TOTAL   0 // entries in the list: the sum of the per-body counts. Written by the scan
#define AVBD_LIST_CONTROL_BROKEN  1 // joints and springs left out because they are fractured
#define AVBD_LIST_CONTROL_INVALID 2 // records or manifolds left out because they name a body slot that is not in use (or too many contacts)
#define AVBD_LIST_CONTROL_WORDS   4

// Threads of the scan kernel: one group scans the per-body counts.
#define AVBD_LIST_SCAN_GROUP 1024

// Contact records per manifold slot (mirrors ContactLayout.MaxContactsPerPair).
#define AVBD_LIST_CONTACTS_PER_SLOT 8

#endif // MCKRILL_AVBD_LIST_PARAMS_INCLUDED
