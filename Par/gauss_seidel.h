/* gauss_seidel.h - Poisson problem
 *
 */
#ifndef _GAUSS_SEIDEL_H
#define _GAUSS_SEIDEL_H

// define your function prototype here
#ifndef _OPENMP
int
gauss_seidel(double ***, double ***, double ***, int, int, double *);
#else
void
gauss_seidel(double ***,double ***,double ***,int,int);
#endif


void
init(double ***u,double ***u_aux,double ***f,int N,double start_T);

#endif
