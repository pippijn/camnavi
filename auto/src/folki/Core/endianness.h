#ifndef _ENDIANNESS_H
#define _ENDIANNESS_H 1

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

# ifndef BIG_ENDIAN
# define BIG_ENDIAN      0
# endif

# ifndef LITTLE_ENDIAN
# define LITTLE_ENDIAN   1
# endif

#include <stdio.h>

typedef short WORD;

typedef union doubleword {
	float reel;
    int entier;
} DWORD;

typedef union quadword {
	double reel;
    WORD s[4];
    unsigned char uc[8];
} QWORD;

int TestByteOrder(void);

/* Lecture */

short GetBigWord(FILE *fp);
short GetLittleWord(FILE *fp);
int GetLittleDoubleWord(FILE *fp);
QWORD GetBigQuadWord(FILE *fp);

size_t befread (void *ptr, size_t size, size_t nmemb, FILE *stream);
/* ATTENTION: aucun test sur la validite du fichier (stream==0) */

/* Ecriture */

void PutBigWord(short w, FILE *fp);
void PutLittleWord(short w, FILE *fp);
void PutBigDoubleWord(int dw, FILE *fp);
void PutLittleDoubleWord(int dw, FILE *fp);
void PutBigQuadWord(QWORD dw, FILE *fp);

size_t befwrite (const void *ptr, size_t size, size_t nmemb, FILE *stream);
/* ATTENTION: aucun test sur la validite du fichier (stream==0) */

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= */

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* endianness.h */
