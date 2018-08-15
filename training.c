#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define INPUT 784
#define HIDDEN 185
#define OUTPUT 10

//------------------------------------------FILE DIR TRAVERSAL-----------------------------------------------------------

int isFile(const char *path)
{
    struct stat path_stat;
    stat(path, &path_stat);
    return S_ISREG(path_stat.st_mode);
}

int getFileCount(char *root)
	{
		struct dirent *de;  
	    DIR *dr = opendir(root);
	    if (dr == NULL)  // opendir returns NULL if couldn't open directory
	    {
	        printf("Could not open current directory" );
	        return 0;
	    }

	    int count=0;

	    while ((de = readdir(dr)) != NULL){
	    		char* t = de->d_name;
	    		char * filePath =malloc(strlen(root)+strlen(t)+1);
				strcat(filePath, root);
				strcat(filePath, de->d_name);
	    	if(isFile(filePath)&&de->d_name[0]!='.'){
	    		count++;
	    	}
	    }
	    return count;
}

void getFilesList(char* files[],char * root){
	struct dirent *de; 
	DIR *dr = opendir(root);
    if (dr == NULL)  // opendir returns NULL if couldn't open directory
    {
        printf("Could not open current directory" );
        return;
    }

    int c=0;

    while ((de = readdir(dr)) != NULL){
		char* t = de->d_name;
		char * filePath =malloc(strlen(root)+strlen(t)+1);
		strcat(filePath, root);
		strcat(filePath, de->d_name);
		if(isFile(filePath)&&de->d_name[0]!='.'){
				files[c]=(char *)malloc((strlen(de->d_name) + 1) * sizeof(char));
	    		// files[c++]=de->d_name;
	    		strcpy(files[c],de->d_name);
	    		c++;
	    }

	}
}

void swap(char **str1, char **str2)
{
	char * temp = *str1;
	*str1 = *str2;
	*str2 = temp;
}

void shuffle(char*files[],int n){
	int i;
    srand ( time(NULL) );
    for ( i = n-1; i > 0; i--){
        int j = rand() % (i+1);
        swap(&files[i], &files[j]);
    }
}

double sigmoid(double x){
	// printf("%f - %f\n", x,exp(-x));
	return (1.0 / (1.0 + exp(-x)));
}

void hiddenNodeCalculate(double inputNodes[],double w_input[][HIDDEN],double hiddenNodes[]){
	for(int i=0;i<HIDDEN;++i){
		double weighted_sum=0;
		for(int j=0;j<INPUT;++j){
			weighted_sum+=inputNodes[j]*w_input[j][i];
		}
		//apply sigmoid function
		hiddenNodes[i] = sigmoid(weighted_sum);
	}
}

void outputNodeCalculate(double hiddenNodes[],double w_hidden[][OUTPUT],double outputNodes[]){
	for(int i=0;i<OUTPUT;++i){
		double weighted_sum=0;
		for(int j=0;j<HIDDEN;++j){
			weighted_sum+=hiddenNodes[j]*w_hidden[j][i];
		}
		//apply sigmoid function
		outputNodes[i] = sigmoid(weighted_sum);
	}
}

int main(){


	double w_input[INPUT][HIDDEN], delta_w_input[INPUT][HIDDEN];
	double w_hidden[HIDDEN][OUTPUT], delta_w_hidden[HIDDEN][OUTPUT];
	double inputNodes[INPUT];
	double hiddenNodes[HIDDEN], deltaHidden[HIDDEN];
	double outputNodes[OUTPUT], deltaOutput[OUTPUT];
	double desiredOutput[OUTPUT];
	double alpha = 0.1, beta = 0.1;

	memset(delta_w_hidden,0.0,sizeof(delta_w_hidden));
	memset(delta_w_input,0.0,sizeof(delta_w_input));

    //Random weights for input to hidden layer
    for(int i=0;i<INPUT;++i){
    	for(int j=0;j<HIDDEN;++j){
    		w_input[i][j]=(float)rand()/RAND_MAX;
    	}
    }

	//Random weights for hidden to output layer
    for(int i=0;i<HIDDEN;++i){
    	for(int j=0;j<OUTPUT;++j){
    		w_hidden[i][j]=(float)rand()/RAND_MAX;
    	}
    }

    //Traversing all files one by one
	
    char * root = "./trainingMatrices/";

    int fileCount = getFileCount(root);

    printf("%d\n", fileCount);

    char * files[fileCount];

    printf("Reading files......\n");
    getFilesList(files,root);
    printf("Files read!!!!!\n");

    printf("Shuffling files......\n");
    shuffle(files,fileCount);
    printf("Shuffling done!!!!!\n");

    for(int f = 0;f<fileCount;++f){
    	char * filePath =malloc(strlen(root)+strlen(files[f])+1);
		strcat(filePath, root);
		strcat(filePath, files[f]);

		printf("%d - %s\n", f+1 , files[f]);
		FILE * file;
		file = fopen(filePath,"r");
		int c=0,i;
		fscanf(file,"%d",&i);

	    while(!feof(file)){
	    	inputNodes[c++]=i;
	    	fscanf(file,"%d",&i);
	    }
	    fclose(file);
	    
	    for(i=0; i<INPUT; i++){
			inputNodes[i] = (inputNodes[i]) / 255;
		}

	    hiddenNodeCalculate(inputNodes,w_input,hiddenNodes);
	    outputNodeCalculate(hiddenNodes,w_hidden,outputNodes);

	    //Backpropagation
	    int op = files[f][0]-'0';
	    double errtemp;

	    for(i=0; i<OUTPUT ; i++)
			desiredOutput[i] =0;//all outputs 0 except for the actual  desired output
		desiredOutput[op]=1;

	    //OUTPUT Layer
	    for(int i=0; i<OUTPUT; i++){
			errtemp = desiredOutput[i] - outputNodes[i];
			deltaOutput[i] = -errtemp * sigmoid(outputNodes[i]) * (1.0 - sigmoid(outputNodes[i]));
		}

		//HIDDEN Layer
		for(int i=0; i<HIDDEN; i++){
			errtemp = 0.0;
			for(int j=0; j<OUTPUT; j++)
				errtemp += deltaOutput[j] * w_hidden[i][j];
			deltaHidden[i] = errtemp * sigmoid(hiddenNodes[i]) * (1.0 - sigmoid(hiddenNodes[i]));
		}

		// Stochastic gradient descent
		for(int i=0; i<OUTPUT; i++){
			for(int j=0; j<HIDDEN; j++){
				delta_w_hidden[j][i] = alpha * delta_w_hidden[j][i] + beta * deltaOutput[i] * hiddenNodes[j];
				w_hidden[j][i] -= delta_w_hidden[j][i];
			}
		}

		for(int i=0; i<HIDDEN; i++){
			for(int j=0; j<INPUT; j++){
				delta_w_input[j][i] = alpha * delta_w_input[j][i] + beta * deltaHidden[i] * inputNodes[j];
				w_input[j][i] -= delta_w_input[j][i];
			}
		}
    } 

    FILE * file;
	file = fopen("inputWeight.txt","w");
	for(int i=0;i<INPUT;++i){
    	for(int j=0;j<HIDDEN;++j){
    		fprintf(file,"%f ",w_input[i][j]);
    	}
    	fprintf(file,"\n");
    }
    fclose(file);

    FILE * fileHidden;
	fileHidden= fopen("hiddenWeight.txt","w");
	for(int i=0;i<HIDDEN;++i){
    	for(int j=0;j<OUTPUT;++j){
    		fprintf(fileHidden,"%f ",w_hidden[i][j]);
    	}
    	fprintf(fileHidden,"\n");
    }
    fclose(fileHidden);

    printf("Weights written to file!!!!!\n");
    printf("DONE!!!!!\n");


	return 0;
}