/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <float.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>

void
gauss_seidel(double ***u,double ***u_aux,double ***f,int N,int iter_max) {

	int i,j,k;
	int it;
	double h=2.0/(N+1.0);
	double pp=1.0/6.0;

	for (it=0;it<iter_max;it++){
	
		// copy u to u_aux
		#pragma omp for private(i,j,k) collapse(3)
		for (i=1;i<=N;i++){
			for (j=1;j<=N;j++){
				for (k=1;k<=N;k++){
					u_aux[i][j][k]=u[i][j][k];
					//int t_id=omp_get_thread_num();
					//printf("thread = %d, (i,j,k) = (%d,%d,%d), it = %d\n",t_id,i,j,k,it);
				}
			}
		}
		
		// updating u
		#pragma omp for private(i,j,k) ordered(3)
		for (int i=1;i<=N;i++){
			for (int j=1;j<=N;j++){
				for (int k=1;k<=N;k++){
					#pragma omp ordered depend(sink:i-1,j,k) depend(sink:i,j-1,k) depend(sink:i,j,k-1)
					u[i][j][k]=(u[i-1][j][k]+u_aux[i+1][j][k]+u[i][j-1][k]+u_aux[i][j+1][k]+u[i][j][k-1]+u_aux[i][j][k+1]+h*h*f[i][j][k])*pp;
					//int t_id=omp_get_thread_num();
					//printf("thread = %d, (i,j,k) = (%d,%d,%d), it = %d, h = %lf, pp = %lf\n",t_id,i,j,k,it,h,pp);
					#pragma omp ordered depend(source)
				}
			}
		}
		
	}
}
#else
double norm(double ***a,double ***b,int N){
	// function calculates norm between matrices
	double sum=0.0;
	int i,j,k;
	
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
gauss_seidel(double ***u,double ***u_aux,double ***f,int N,int iter_max,double *tol) {

	int i,j,k;
	double d=DBL_MAX;
	int it=0;
	double h=2.0/(N+1.0);
	double pp=1.0/6.0;
	
	while (d>*tol && it<iter_max){
	
		// copy u to u_aux
		for (i=1;i<=N;i++){
			for (j=1;j<=N;j++){
				for (k=1;k<=N;k++){
					u_aux[i][j][k]=u[i][j][k];
				}
			}
		}
		
		// updating u
		for (i=1;i<=N;i++){
			for (j=1;j<=N;j++){
				for (k=1;k<=N;k++){
					u[i][j][k]=(u[i-1][j][k]+u_aux[i+1][j][k]+u[i][j-1][k]+u_aux[i][j+1][k]+u[i][j][k-1]+u_aux[i][j][k+1]+h*h*f[i][j][k])*pp;
				}
			}
		}
		d=norm(u,u_aux,N);
		it++;
	}
	*tol=d;
	return(it);
}
#endif
