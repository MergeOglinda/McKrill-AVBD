#ifndef MCKRILL_AVBD_COLORING_PARAMS_INCLUDED
#define MCKRILL_AVBD_COLORING_PARAMS_INCLUDED

// Constants of stage C (graph coloring). Mirrored by Runtime/Constraints/AvbdGraphColoring.cs; a test checks the two against each other.

// Most colors a graph may use. A body that would need more is given the last color and the overflow word is set.
#define AVBD_COLOR_MAX 64

// Value of the graph-color field (Meta.state.x) of a solved body that has no color yet. A body that is not solved keeps AVBD_INVALID.
#define AVBD_COLOR_PENDING 0xFFFFFFFEu

// Layout of the work buffer, in 32-bit words: control words, then the body count of every color, then the next free entry of every color
// (the bucket cursors), then the start of every color in the bucket array.
#define AVBD_COLOR_CONTROL_WORDS 4   // 0 = bodies still without a color after the last round, 1 = bodies that needed more than AVBD_COLOR_MAX colors,
                                     // 2 = number of colors (the highest color + 1), 3 unused
#define AVBD_COLOR_COUNT_WORD 2
#define AVBD_COLOR_COUNTS_OFFSET 4
#define AVBD_COLOR_CURSORS_OFFSET 68
#define AVBD_COLOR_STARTS_OFFSET 132
#define AVBD_COLOR_WORK_WORDS 196

// Indirect dispatch arguments of the primal sweep: three words (groups x, y, z) per color, written by ColorPlan. A color of more than
// AVBD_COLOR_ROW_GROUPS thread groups is laid out in rows of that many groups (x = the row width, y = the number of rows), so the thread
// with SV_DispatchThreadID (x, y) handles entry y * AVBD_COLOR_ROW_GROUPS * AVBD_GROUP_SIZE + x of its color; the last row may overshoot.
#define AVBD_COLOR_ARGS_STRIDE 12    // bytes per color
#define AVBD_COLOR_ROW_GROUPS 65535u

// Priority order of the coloring.
#define AVBD_COLOR_ORDER_SLOT   0 // a higher slot colors first: the same coloring as the reference solver's sequential greedy
#define AVBD_COLOR_ORDER_HASHED 1 // by a hash of the slot: far fewer rounds on long chains, a different (equally valid) coloring

#endif // MCKRILL_AVBD_COLORING_PARAMS_INCLUDED
