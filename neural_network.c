# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include "mpi.h"

//================================================
// PARAMETERS
#define PI 3.14159265358979323846
// Activation
#undef RELU // He initialization
#define TANH // Xavier initialization
// #define RELU 
// #undef TANH
//================================================


//================================================
// FUNCTIONS

void read_file(char * file_name, 
                int * n_layers, int * n_neurons, 
                int * inputs_dim, int * output_dim,
                double ** inputs, double ** output_gt){
    // Read a file defining a NN with the following structure
    // Number of inputs
    // Inputs
    // Number of layers
    // Number of neurons per layer
    // Number of outputs
    // Desidered outputs
    FILE *fp;
    int i, success=1;
    double *in, *out;
    
    fp = fopen(file_name, "r");
    if( fp != NULL ){

        // Read inputs dimension
        if(fscanf(fp, "%d", inputs_dim) <=0){
            printf("Error reading the inputs dimension\n");
            success = 0;
        } 

        // Read the inputs
        in = (double *) malloc(*inputs_dim * sizeof(double));
        for(i=0; i<*inputs_dim; i++){
            if(fscanf(fp, "%lf", &in[i]) <=0){
                printf("Error reading the inputs\n");
                success = 0;
                break;
            }
        }
        *inputs=in;

        // Read the number of layers
        if(fscanf(fp, "%d", n_layers) <=0){
            printf("Error reading the number of layers\n");
            success = 0;
        } 

        // Read the number of neurons per layer
        if(fscanf(fp, "%d", n_neurons) <=0){
            success = 0;
        } 

        // Read the output_gt dimension
        if(fscanf(fp, "%d", output_dim) <=0){
            printf("Error reading the output_gt dimension\n");
            success = 0;
        } 

        // Read the output_gt
        out = (double *) malloc(*output_dim * sizeof(double));
        for(i=0; i<*output_dim; i++){
            if(fscanf(fp, "%lf", &out[i]) <=0){
                printf("Error reading the output_gt\n");
                success = 0;
                break;
            }
        }
        *output_gt=out;

    } else {
        printf("Error reading the file\n");
        success = 0;
    }
    
    
    if (! success){
        exit(2);
    } else{
        fclose(fp);
    }
}

double relu(double num){
    if(num>0){
        return num;
    }else{
        return 0;
    }
}

double activation(double num){
    double res;

    #ifdef TANH
        res = tanh(num);
    #elif defined(RELU)
        res = relu(num);
    #endif

    return res;
}

double uniform_sample (double minV, double maxV){
    double U_0_1;
    U_0_1 = ( (double)(rand()) *1.0 )/( (double)(RAND_MAX) );
    return ((maxV - minV) * U_0_1 + minV);
}

double normal_sample(double mu, double sigma) {
   // return a normally distributed random value
   double v1=uniform_sample(0, 1);
   double v2=uniform_sample(0, 1);
   return (cos(2*PI*v2)*sqrt(-2.*log(v1)) * sigma + mu);
}

double xavier_init(int n_prev, int n_next){

    double bound = sqrt(6.0/ (double) (n_prev + n_next));

    return uniform_sample(-bound, bound);
}

double he_init(int n_prev){
    return (normal_sample(0,1) * sqrt(2.0 /(double) n_prev));
}

double rand_init(){
    return uniform_sample(-1,1);
}

double initialize(int n_prev, int n_next){
    double res;
    #ifdef TANH
        res = xavier_init(n_prev, n_next); // Xavier init
    #elif defined(RELU)
        res = he_init(n_prev); // He=Kaming init
    #else
        res = rand_init(); // U[-1, 1]
    #endif
    return res;
}
//================================================

int main ( int argc , char * argv [ ]) {
    //================================================
    // SETUP
    //------------------------------------------------

    //Variables 
    int myId, numProcs, i, j, l;
    int n_layers, n_neurons, neuronsL, inputs_dim, output_dim;
    double *nn_wL, *nn_bL, *inputs, *output, *output_gt, weight, bias;
    int info[4]; // [inputs_dim, n_neurons, n_layers, output_dim]

    // MPI initialization
    MPI_Init( & argc, & argv );
    MPI_Comm_size( MPI_COMM_WORLD, & numProcs );
    MPI_Comm_rank( MPI_COMM_WORLD, & myId );

    // Check the arguments
    if(myId == 0 && argc != 2){
        printf("Incorrect number of parameters, expected 1 and got %d\n", argc-1);
        exit(1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Only proc_0 reads the file
    if(myId == 0){
        read_file(argv[1], &n_layers, &n_neurons, 
                    &inputs_dim, &output_dim, &inputs, &output_gt);

        // Prepare the info package
        info[0] = inputs_dim;
        info[1] = n_neurons;
        info[2] = n_layers;
        info[3] = output_dim;
    }

    // Share the information to all the procs
    MPI_Bcast(info, 3, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (myId != 0){
        inputs_dim = info[0];
        n_neurons = info[1];
        n_layers=info[2];
        output_dim = info[3];

        // Allocate memory to receive the inputs
        inputs = (double *)  malloc( inputs_dim * sizeof( double ) );
    }
    neuronsL = (int) (n_neurons/numProcs);

    if(myId == 0){
        // Check the information is well read 
        printf("=========================================\n");
        printf("Information read in process %d\n", myId);
        printf("=========================================\n");
        printf("Inputs dimension: \t%d\n", inputs_dim);
        printf("Inputs: \t[%lf", inputs[0]);
        for(i=1; i<inputs_dim; i++){
            printf(", %lf", inputs[i]);
        }
        printf("]\n");
        printf("Number of layers: \t%d\n", n_layers);
        printf("Number of neurons per layer:\t%d\n", n_neurons);
        printf("output_gt dimension: \t%d\n", output_dim);
        printf("Expected output_gt: \t[%lf", output_gt[0]);
        for(i=1; i<output_dim; i++){
            printf(", %lf", output_gt[i]);
        }
        printf("]\n");
        printf("Number of neurons for each process: \t%d\n", neuronsL);
        printf("Number of parameters: \t%d\n", (n_layers*n_neurons+ output_dim)*2);
        printf("=========================================\n");
        printf("\n");
    }

    // Initialize each layer differently, also depending on the activation
    nn_wL = (double *)  malloc( (n_layers * neuronsL + output_dim) * sizeof( double ) );
    nn_bL = (double *)  malloc( (n_layers * neuronsL + output_dim) * sizeof( double ) );
    srand ( myId + (int) MPI_Wtime()) ;

    // Initialization 
    // printf("First layer:\n");
    for(i=0; i<neuronsL; i++){
        nn_wL[i] = initialize(inputs_dim, n_neurons);
        nn_bL[i] = 0;
        // printf("%lf\n", nn_wL[i]);
    }
    // printf("\n");

    // printf("Hidden layers:\n");
    for(i=neuronsL; i<(n_layers * neuronsL);i++){
        nn_wL[i] = initialize(n_neurons, n_neurons); 
        nn_bL[i] = 0; 
        // printf("%lf\n", nn_wL[i]);
    }
    // printf("\n");

    // printf("Output layer:\n");
    for(i=(n_layers * neuronsL); i<(n_layers * neuronsL + output_dim);i++){
        nn_wL[i] = initialize(n_neurons, output_dim); 
        nn_bL[i] = 0;
        // printf("%lf\n", nn_wL[i]);
    }
    // printf("\n");
    

    // Initialize the outputs for each layer
    double output_layers[n_layers][n_neurons];
    double output_local[neuronsL];
    output = (double *)  malloc( output_dim * sizeof( double ) );

    // Share the inputs to all the procs
    MPI_Bcast(inputs, inputs_dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //================================================
    // Uncoment to ensure the correct order when printing, results are not affected
    //MPI_Barrier(MPI_COMM_WORLD); 
    //================================================
    // FOWARD
    //------------------------------------------------

    // First layer and inputs
    for(i=0; i<neuronsL; i++){
        output_local[i]=0.0;
        // Get the weight and bias from the neuron
        weight = nn_wL[i];
        bias = nn_bL[i];

        for(j=0; j<inputs_dim; j++){
            output_local[i] += weight * inputs[j];
        }
        
        output_local[i] = activation(output_local[i]+bias);
        //printf("Proc_%d: %lf\n", myId, output_local[i]);
    }
    MPI_Allgather(output_local, neuronsL, MPI_DOUBLE,
                    output_layers[0], neuronsL, MPI_DOUBLE, MPI_COMM_WORLD);

    ///////////////////////////////////////////////
    // DEBUG PRINT
    // if(myId==1){
    //     printf("First Layer\n");
    //     for(i=0;i<n_neurons;i++){
    //         printf("%lf\n", output_layers[0][i]);
    //     }
    // }
    ///////////////////////////////////////////////

    // Hidden layers (layer 0 are the inputs)
    for(l=1; l<n_layers; l++){
        for(i=0; i<neuronsL; i++){
            output_local[i]=0.0;
            // Get the weight and bias from the neuron
            weight = nn_wL[l*neuronsL+i];
            bias = nn_bL[i];

            for(j=0; j<n_neurons; j++){
                output_local[i] += weight * output_layers[l-1][j];
            }
            output_local[i] = activation(output_local[i]+bias);
            // printf("Proc_%d: %lf\n", myId, output_local[i]);
        }
        MPI_Allgather(output_local, neuronsL, MPI_DOUBLE,
                    output_layers[l], neuronsL, MPI_DOUBLE, MPI_COMM_WORLD);

        ///////////////////////////////////////////////
        // DEBUG PRINT
        // if(myId==1){
        //     printf("Layer %d\n", l);
        //     for(i=0;i<n_neurons;i++){
        //         printf("%lf\n", output_layers[l][i]);
        //     }
        // }
        ///////////////////////////////////////////////
    }

    // Output layer
    for(i=0; i<output_dim; i++){
        output_local[i]=0.0;
        // Get the weight and bias from the neuron
        weight = nn_wL[(n_layers*neuronsL)+i];
        bias = nn_bL[(n_layers*neuronsL)+i];

        for(j=0; j<n_neurons; j++){
            output_local[i] += weight * output_layers[n_layers-1][j];
        }
        output_local[i] = output_local[i]+bias;
        // printf("Proc_%d: %lf\n", myId, output_local[i]);
    }

    MPI_Allgather(output_local, neuronsL, MPI_DOUBLE,
                    output, neuronsL, MPI_DOUBLE, MPI_COMM_WORLD);

    //================================================

    if(myId == 0){
        printf("Result from the foward pass: [");
        printf("%lf", output[0]);
        for(i=1; i<output_dim; i++){
            printf(", %lf", output[i]);
        }
        printf("]\n");
    }
    
    exit(0);
}
