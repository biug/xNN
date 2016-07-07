#ifndef MACROS_H
#define MACROS_H

#define TOKEN_BEGIN			"-BEGIN-"
#define TOKEN_TAIL			"-TAIL"
#define TOKEN_START_CODE	2

#define	MAX_SENTENCE_SIZE	256
#define	MAX_SENTENCE_BITS	8

#define MAX_LABEL_ID			256
#define MAX_LABEL_ID_BITS		8
#define LEFT_LABEL_ID(LI)		((LI) >> MAX_LABEL_ID_BITS)
#define RIGHT_LABEL_ID(LI)		((LI) & (MAX_LABEL_ID - 1))
#define ENCODE_LABEL_ID(LL,LR)	(((LL) << MAX_LABEL_ID_BITS) | (LR))

#define IS_LEFT_LABEL(L)	(L.find("left") == 0)
#define IS_RIGHT_LABEL(L)	(L.find("right") == 0)
#define IS_TWOWAY_LABEL(L)	(L.find("twoway") == 0)

#define ENCODE_LEFT_LABEL(L)		("left" + L)
#define ENCODE_RIGHT_LABEL(L)		("right" + L)
#define ENCODE_TWOWAY_LABEL(L1,L2)	("twoway" + L1 + "||" + L2)

#define DECODE_LEFT_LABEL(L)			(L.substr(strlen("left")))
#define DECODE_RIGHT_LABEL(L)			(L.substr(strlen("right")))
#define DECODE_TWOWAY_LEFT_LABEL(L)		(L.substr(strlen("twoway"), L.find("||") - strlen("twoway")))
#define DECODE_TWOWAY_RIGHT_LABEL(L)	(L.substr(L.find("||") + strlen("||")))

#define	GLOBAL_EPSILON		1e-10

#define	SGD_THRESHOLD		1e-4
#define ADAGRAD_THRESHOLD	1e-8

#define	REGULAR_LAMDA		1e-8

#define	SGD_ALPHA			1e-2
#define	SGD_MOMENTUM		0.95

#define	ADAGRAD_ALPHA		1e-3
#define ADAGRAD_EPSILON		1e-8

#endif