/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <float.h>
#include <stdio.h>


#ifdef _OPEN_MP
#include <omp.h>
#endif

double norm(double ***a,double ***b,int N){
	// function calculates norm between arrays
	double sum=0.0;
	int i,j,k;

	#pragma omp taskloop default(none) shared(a,b,N) private(i,j,k) reduction(+:sum) collapse(3)
	for (i=1;i<=N;i++){
		for (j=1;j<=N;j++){
			for (k=1;k<=N;k++){
				double x=a[i][j][k];
				double y=b[i][j][k];
				sum+=(x-y)*(x-y);
			}
		}
	}
	return sqrt(sum);
}

int
jacobi(double ***u,double ***u_aux,double ***f,int N,int iter_max,double *tol) {

	int i,j,k;
	double h=2.0/(N+1.0);
	double pp=1.0/6.0;
	double d=DBL_MAX;
	int it=0;
	
	while (it<iter_max){
		#pragma omp task
		{
		// copy u to u_aux
		#pragma omp taskloop collapse(3)
		for (i=1;i<=N;i++){
			for (j=1;j<=N;j++){
				for (k=1;k<=N;k++){
					u_aux[i][j][k]=u[i][j][k];
				}
			}
		}
		
		// updating u
		#pragma omp taskloop collapse(3)
		for (i=1;i<=N;i++){
			for (j=1;j<=N;j++){
				for (k=1;k<=N;k++){
					u[i][j][k]=(u_aux[i-1][j][k]+u_aux[i+1][j][k]+u_aux[i][j-1][k]+u_aux[i][j+1][k]+u_aux[i][j][k-1]+u_aux[i][j][k+1]+h*h*f[i][j][k] )*pp;
				}
			}
		}
		
		} /* end of omp task */
		#pragma omp taskwait
		d=norm(u,u_aux,N);
		it++;
	}
	*tol=d;
	return it;
}
