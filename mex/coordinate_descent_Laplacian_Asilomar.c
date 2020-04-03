/*=========================================================
 *
 *Authors - Eduardo Pavez <eduardo.pavez.carvelli@gmail.com, pavezcar@usc.edu>
 *Copyright Eduardo Pavez,  University of Southern California
 *Los Angeles, USA, 04/02/2020
 *This function implements the algorithm to learn a combinatorial laplacian proposed, see paper for details
 *Pavez, Eduardo, and Antonio Ortega. "An Efficient Algorithm for Graph Laplacian Optimization Based on Effective Resistances."
 *2019 53rd Asilomar Conference on Signals, Systems, and Computers. IEEE, 2019.
 * 
 *=======================================================*/

#if !defined(_WIN32)
#define dsyr dsyr_
#endif

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include "mex.h"
#include "blas.h"
#include "lapack.h"
//

void randperm(mwSize *a, mwSize n){
    // random permutation of elements of array a
    mwSize i,j;
    mwSize aux;
    
    for(i=0;i<n;i++){
        j = rand()%n;
        aux  = a[i];
        a[i] = a[j];
        a[j] = aux;
    }
}

double FrobeniusNorm(double *X, mwSize m, mwSize n){
    double out = 0.0;
    mwSize i;
    for(i=0;i<m*n;i++){
        out = out + X[i]*X[i];
    }
    return out;
}
double compute_objective_initial_tree(  double *edm,
        double *w,
        mwSize n,
        mwSize m){
    
    double value;
    double wcurr;
    
    double zero = 0.0;
    mwSize one = 1;
    double n_double = (double)n;
    mwSize  iter;
    value = -log(n_double) + n_double - 1.0; // the -log(n) term from the log determinant, and the n-1 = \sum_e w_e r_e
    for(iter=0;iter<m;iter++){
        wcurr = w[iter];
        if( wcurr>zero){
            value = value + (-1.0)*log(wcurr);
        }
    }
    return value;
}

//implementes coordinate minimization with various edge selection rules
// cyclic, random, and proximal gauss-southwell rule 
int solver_coordinate_minimization(  double *edm,
        double *edge,
        double *w,
        double *Sigma,
        mwSize n,
        mwSize m,
        double *obj,
        double tol,
        mwSize max_epoch,
        mwSize rule){
    
    double wnew, wold, delta_w, delta_obj;
    double alpha;
    double obj_prev, obj_curr;
    double *x;
    double Sij, Sii, Sjj, effRij;
    double Ski, Skj;
    double max_val;
    double curr_val;
    
    mwSize weight_iteration, iter,  i, j, k, edge_iter, iteration; //
    mwSize max_comparisons;
    mwSize *permutation;
    double very_small = 0.0;
    char *uplo="U";
    mwSize one = 1;
    
    x =   mxCalloc(n,sizeof(double));
    if(rule == 1){
        permutation = mxCalloc(m,sizeof(mwSize));
        for(i=0;i<m;i++){
            permutation[i]=i;
        }
    }
    
    obj_prev = compute_objective_initial_tree(  edm, w, n, m);
    delta_obj = obj_prev;
    obj[0] = delta_obj;
    mexPrintf("initialization | \t obj=%f  \n", delta_obj );
        
    for(weight_iteration=0;weight_iteration<max_epoch*m;weight_iteration++){
        // rule == 0  cyclic, return iter
        // rule == 1  random uniform, return rand()%n +1
        // rule == 2  PGS largest generalized gradient (delta_w)
        // rule == 2  largest obj func descent
        if(rule==1 && weight_iteration%m == 0){
            randperm(permutation,  m);
        }
        
        // low complexity O(1) edge selection rules
        if(rule == 0 || rule == 1){
            switch (rule){
                case 0:
                    iter = weight_iteration%m; // cyclic rule
                    break;
                case 1:
                    iter =rand()%m;// uniform random in {0,1,...,m-1}, sampling with replacement
                    iter = permutation[iter];// this is sampling without replaement
                    break;
            }
        }
        // higher complexity O(m) edge selection rules (greedy)
        else{
            // for the greedy case, below will find the best edge
            max_val =0.0;
            max_comparisons = m;
            for(iteration =0; iteration < max_comparisons; iteration++){
                    
                edge_iter = iteration;
                //pick edge indices, substract 1 to go from matlab indexing to C indexing
                i = (mwSize)edge[edge_iter] - one;        //edge is m x 2 matrix
                j = (mwSize)edge[edge_iter + m] - one;
                if(i==j){
                    return 2;
                }
                // Since Sigma is symmetric, will only use upper triangular part,
                //so swap indices in case the edge is below diagonal
                if(i>j){
                    k = i;
                    i = j;
                    j = k;
                }
                // extract sub-matrix Se = [Sii, Sij ; Sij, Sjj]
                Sii    = Sigma[i + i*n];
                Sjj    = Sigma[j + j*n];
                Sij    = Sigma[i + j*n];
                //effective resistance
                effRij = Sii + Sjj - 2.0*Sij;
                // compute closed form update
                wold = w[edge_iter];
                wnew = 1/edm[edge_iter] - 1/effRij + wold;
                //treshold if negative
                if(wnew<0.0){
                    wnew=0.0;
                }
                delta_w = wnew-wold;
                //now check if this edge is good enough
                switch (rule){
                    case 2:// largest generalized gradient (delta_w)
                        curr_val = fabs(delta_w);
                        break;
                    case 3:// largest obj function decrease (delta_obj)
                        curr_val = fabs(-log(1.0+delta_w*effRij) + delta_w*edm[edge_iter]);
                        break;
                }
                // after curr_val has been computed, check if this is an improvement over max_val
                if(curr_val>max_val){
                    max_val = curr_val;
                    iter = edge_iter;// this is the optimal edge
                }
            }
        }      
// now that the optimal edge has been chosen, do actual update
        //pick edge indices, substract 1 to go from matlab indexing to C indexing
        i = (mwSize)edge[iter] - one;        //edge is m x 2 matrix
        j = (mwSize)edge[iter + m] - one;
        if(i==j){
            return 2;
        }
        // Since Sigma is symmetric, will only use upper triangular part,
        //so swap indices in case the edge is below diagonal
        if(i>j){
            k = i;
            i = j;
            j = k;
        }
        // extract sub-matrix Se = [Sii, Sij ; Sij, Sjj]
        Sii    = Sigma[i + i*n];
        Sjj    = Sigma[j + j*n];
        Sij    = Sigma[i + j*n];
        //
        effRij = Sii + Sjj - 2.0*Sij;
        // compute closed form update
        wold = w[iter];
        wnew = 1/edm[iter] - 1/effRij + wold;
        if(wnew<0.0){
            wnew=0.0;
        }
        delta_w = wnew-wold;
        //only  update if delta is big enough
        if(fabs(delta_w)>very_small){
            
            w[iter]=wnew;// update weight
            
            //update Sigma
            //x is computed using upper triangular part of Sigma so
            //do careful computation of i-th and j-th column of Sigma
            for(k=0;k<n;k++){
                //compute Sigma(k,i)
                if(k<i){
                    Ski = Sigma[k + i*n];
                }
                else{
                    Ski = Sigma[i + k*n];
                }
                //Compute Sigma(k,j)
                if(k<j){
                    Skj = Sigma[k + j*n];
                }
                else{
                    Skj = Sigma[j + k*n];
                }
                x[k] = Ski - Skj;
            }
            alpha = -delta_w/(1.0 + effRij*delta_w);
            
            dsyr(uplo,&n, &alpha, x,  &one, Sigma, &n);//the  rank 1 update
            delta_obj = delta_obj -log(1.0+delta_w*effRij) + delta_w*edm[iter];// update on obj function  
        }
        
        // mexPrintf("chosen edge=%d \n",iter);
        // if 1 epoch has passed, then save value
        if (weight_iteration%m == 0 && weight_iteration >0){
            obj[weight_iteration/m] = delta_obj;
            obj_curr = delta_obj;
            // mexPrintf("epoch=%d | \t obj=%f | delta=%E \n",weight_iteration/m, delta_obj,obj_curr - obj_prev );
            if(fabs(obj_curr - obj_prev)<tol){
                // mexPrintf("%f  \n",errF);
                break;
            }
            obj_prev = obj_curr;
        }   
    }
    if(rule==1){
        mxFree(permutation);
    }
    mexPrintf("epoch=%d | \t obj=%f | delta=%E \n",weight_iteration/m, delta_obj,obj_curr - obj_prev );
    mxFree(x);
    return 1;
}
//[wout, obj] = coordinate_descent_Laplacian_Asilomar(edm,edge, w,S, tol, niter, rule ), matlab call
// input/output mexFunction
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *edm, *edge, *w_in, *Sigma_in; /* pointers to input  matrices*/
    double *w_out, *Sigma_out, *obj;    /* pointers to output matrices*/
    mwSize n, m, max_epoch, rule,randomized_search;   /*input parameters*/
    double tol;                           /* tolerance for convergence*/
    
    //////////////////
//get dimensions
    m = mxGetM(prhs[0]); //number of rows of edm values is number of edges
    n = mxGetM(prhs[3]); // number of rows of input S matrix
    
    //mexPrintf("%d  \n",m);
    
    //check if dimensions of input match
    if (m != mxGetM(prhs[1]) ||  m != mxGetM(prhs[2]) ) {
        mexErrMsgIdAndTxt("Input:dimmismatch","edm, w_in,  edge must have equal number of rows.");
    }
    if(1 != mxGetN(prhs[0]) || 1 != mxGetN(prhs[2]) ){
        mexErrMsgIdAndTxt("Input:dimmismatch","edm, w_in,   must have 1 column.");
    }
    if(2 != mxGetN(prhs[1])){
        mexErrMsgIdAndTxt("Input:dimmismatch","edge   must have 2 column.");
    }
    if(n != mxGetN(prhs[3])){
        mexErrMsgIdAndTxt("Input:dimmismatch","S must be square");
    }
    //just get the pointers to input data
    edm   = mxGetPr(prhs[0]); /* first input matrix */ // edm
    edge  = mxGetPr(prhs[1]); /* second input matrix */
    w_in  = mxGetPr(prhs[2]);
    Sigma_in  = mxGetPr(prhs[3]);
    tol       = mxGetScalar(prhs[4]);
    max_epoch  = (mwSize)mxGetScalar(prhs[5]);/*max num iterations, num of times each edge weight is updated*/
    rule  = (mwSize)mxGetScalar(prhs[6]);/*max num iterations, num of times each edge weight is updated*/
    //randomized_search  = (mwSize)mxGetScalar(prhs[7]);/* */
    
    //pointers to output data
    plhs[0] = mxCreateDoubleMatrix(m, 1, mxREAL);     // output weights
    plhs[1] = mxCreateDoubleMatrix(max_epoch, 1, mxREAL);
    w_out = mxGetPr(plhs[0]);
    obj = mxGetPr(plhs[1]);
    //copy inputs onto temporary arrays
    Sigma_out= (double*)calloc( n*n, sizeof(double));
    memcpy((void*)w_out,  (void*)w_in, m*sizeof(double)); //copies input weight into output weight vector
    memcpy((void*)Sigma_out,  (void*)Sigma_in, n*n*sizeof(double)); //copies input cov into output cov
    
    
//computational routine
    
    int success = solver_coordinate_minimization(  edm,
            edge,
            w_out,
            Sigma_out,
            n,
            m,
            obj,
            tol,
            max_epoch,
            rule);
    
    free(Sigma_out);
    
    if(success == 2){
        mexErrMsgIdAndTxt("Input:edgeerror","edge includes a self loop");
        
    }
}
