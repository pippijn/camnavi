#include "endianness.h"

int TestByteOrder(void)
{
  short int word = 0x0001;
  char *byte = (char *) &word;
  return(byte[0] ? LITTLE_ENDIAN : BIG_ENDIAN);
}

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= */

/* Lecture */
WORD GetBigWord(FILE *fp)
{
  register WORD w;
  w =  (WORD) (fgetc(fp) & 0xFF);
  w = ((WORD) (fgetc(fp) & 0xFF)) | (w << 0x08);
  return(w);
}
       
WORD GetLittleWord(FILE *fp)
{
  register WORD w;
  w =  (WORD) (fgetc(fp) & 0xFF);
  w |= ((WORD) (fgetc(fp) & 0xFF) << 0x08);
  return(w);
}
       
int GetBigDoubleWord(FILE *fp)
{
  register int dw;
  dw =  (int) (fgetc(fp) & 0xFF);
  dw = ((int) (fgetc(fp) & 0xFF)) | (dw << 0x08);
  dw = ((int) (fgetc(fp) & 0xFF)) | (dw << 0x08);
  dw = ((int) (fgetc(fp) & 0xFF)) | (dw << 0x08);
  return(dw);
}
       
QWORD GetBigQuadWord(FILE *fp)
{
/* pas termine */

  QWORD qw;
/*  int dw0,dw1;
  dw0 =  GetBigDoubleWord(fp);
  dw1 =  GetBigDoubleWord(fp);
  qw.dw[0] = dw1; qw.dw[1] = dw0;*/
  return(qw);
}
       
int GetLittleDoubleWord(FILE *fp)
{
  register int dw;
  dw =  (int) (fgetc(fp) & 0xFF);
  dw |= ((int) (fgetc(fp) & 0xFF) << 0x08);
  dw |= ((int) (fgetc(fp) & 0xFF) << 0x10);
  dw |= ((int) (fgetc(fp) & 0xFF) << 0x18);
  return(dw);
}
       
size_t befread (void *ptr, size_t size, size_t nmemb, FILE *stream) {
	int i;
    WORD* pW; DWORD* pDW;
    
    i=nmemb;
    if (nmemb==1)
    	switch (size) {
        case 2:
        	pW = ptr;
        	if (!feof(stream)) *pW = GetBigWord(stream); else i=0;
            break;
        case 4:
        	pDW = ptr;
        	if (!feof(stream)) pDW->entier = GetBigDoubleWord(stream); else i=0;
            break;
        default:
        	i = fread(ptr, size, nmemb, stream);
        }
    else
    	switch (size) {
        case 2:
        	for (i=0,pW=ptr;(i<nmemb) && (!feof(stream));i++,pW++) *pW=GetBigWord(stream);
            if (feof(stream)) i--;
            break;
        case 4:
        	for (i=0,pDW=ptr;(i<nmemb) && (!feof(stream));i++,pDW++) pDW->entier=GetBigDoubleWord(stream);
            if (feof(stream)) i--;
            break;
        default:
        	i = fread(ptr, size, nmemb, stream);
        }
	return(i);
/*    return (feof(stream))?0:nmemb; */
}


/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= */


/* Ecriture */

void PutBigWord(short w, FILE *fp)
{
  fputc((w >> 0x08) & 0xFF, fp);
  fputc(w & 0xFF, fp);
}
       
void PutLittleWord(short w, FILE *fp)
{
  fputc(w & 0xFF, fp);
  fputc((w >> 0x08) & 0xFF, fp);
}
       
/*void PutBigDoubleWord(int *dw, FILE *fp) */ /* pour PutBigDoubleWord((int*)pFloat++, fc); */
void PutBigDoubleWord(int dw, FILE *fp)
{
  fputc((dw >> 0x18) & 0xFF, fp);
  fputc((dw >> 0x10) & 0xFF, fp);
  fputc((dw >> 0x08) & 0xFF, fp);
  fputc(dw & 0xFF, fp);
}
       
void PutLittleDoubleWord(int dw, FILE *fp)
{
  fputc(dw & 0xFF, fp);
  fputc((dw >> 0x08) & 0xFF, fp);
  fputc((dw >> 0x10) & 0xFF, fp);
  fputc((dw >> 0x18) & 0xFF, fp);
}

void PutBigQuadWord(QWORD qw, FILE *fp)
{
/* pas termine */

/*  PutBigDoubleWord(qw.dw[0], fp);
  PutBigDoubleWord(qw.dw[1], fp);*/
/*  fputc((qw.lg >> 0x38) & 0xFF, fp);
  fputc((qw.lg >> 0x30) & 0xFF, fp);
  fputc((qw.lg >> 0x28) & 0xFF, fp);
  fputc((qw.lg >> 0x20) & 0xFF, fp);
  fputc((qw.lg >> 0x18) & 0xFF, fp);
  fputc((qw.lg >> 0x10) & 0xFF, fp);
  fputc((qw.lg >> 0x08) & 0xFF, fp);
  fputc(qw.lg & 0xFF, fp);*/
}
 
size_t befwrite (const void *ptr, size_t size, size_t nmemb, FILE *stream) {
	int i;
    WORD* pW; DWORD* pDW;
    
/*    printf("size:%d nmemb: %d sizeof(DWORD):%d sizeof(int):%d\n", size, nmemb, sizeof(DWORD), sizeof(int));*/
	i=nmemb;
    if (nmemb==1)
    	switch (size) {
        case 2:
        	pW = (WORD*)ptr;
        	PutBigWord(*pW, stream);
            break;
        case 4:
        	pDW = (DWORD*)ptr;
        	PutBigDoubleWord(pDW->entier, stream);
            break;
       default:
        	i=fwrite(ptr, size, nmemb, stream);
        }
    else
    	switch (size) {
        case 2:
        	for (i=0,pW=(WORD*)ptr;i<nmemb;i++,pW++) PutBigWord(*pW, stream);
            break;
        case 4:
        	for (i=0,pDW=(DWORD*)ptr;i<nmemb;i++,pDW++) PutBigDoubleWord(pDW->entier, stream);
            break;
        default:
        	i=fwrite(ptr, size, nmemb, stream);
        }
    return (i);
}


/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= */
