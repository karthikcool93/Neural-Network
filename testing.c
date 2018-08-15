#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>
#include <stdint.h>

#define INPUT 784
#define HIDDEN 185
#define OUTPUT 10

double sigmoid(double x){
	// printf("%f - %f\n", x,exp(-x));
	return(1.0 / (1.0 + exp(-x)));
}

int isFile(const char *path)
{
    struct stat path_stat;
    stat(path, &path_stat);
    return S_ISREG(path_stat.st_mode);
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

	char * root = "./testSet/";

	struct dirent *de;

    DIR *dr = opendir(root);
    if (dr == NULL)  // opendir returns NULL if couldn't open directory
    {
        printf("Could not open current directory" );
        return 0;
    }

    FILE * file;
    file = fopen("./inputWeight.txt","r");
	float ip;
	fscanf(file,"%f",&ip);
	int row=0,col=0;
	while(!feof(file)){
		if(col==HIDDEN){
			row++;
			col=0;
		}
		w_input[row][col]=ip;
		fscanf(file,"%f",&ip);
		col++;
	}
	fclose(file);

	file = fopen("./hiddenWeight.txt","r");
	fscanf(file,"%f",&ip);
	row=0,col=0;
	while(!feof(file)){
		if(col==OUTPUT){
			row++;
			col=0;
		}
		w_hidden[row][col]=ip;
		fscanf(file,"%f",&ip);
		col++;
	}
	fclose(file);

	int misses=0,hits=0;

	printf("Test case\tExpected output\t\tComputed output\n");
	printf("---------------------------------------------------------\n");
	int count = 0;

    while ((de = readdir(dr)) != NULL){
    		char* t = de->d_name;
    		char * filePath =malloc(strlen(root)+strlen(t)+1);
			strcat(filePath, root);
			strcat(filePath, de->d_name);
    	if(isFile(filePath)&&de->d_name[0]!='.'){

			file = fopen(filePath,"r");
			int c=0,value;
			fscanf(file,"%d",&value);
		    while(!feof(file)){
		    	inputNodes[c++]=value;
		    	fscanf(file,"%d",&value);
		    }
		    fclose(file);

		    for(int i=0; i<INPUT; i++){
				inputNodes[i] = (inputNodes[i]) / 255;
			}

			int desired_output = de->d_name[0]-'0';

			hiddenNodeCalculate(inputNodes,w_input,hiddenNodes);
			outputNodeCalculate(hiddenNodes,w_hidden,outputNodes);

			double m=0.0;
			int m_i=0;
		    for(int i=0;i<OUTPUT;++i){
		    	if(outputNodes[i]>m){
		    		m=outputNodes[i];
		    		m_i=i;
		    	}
		    }
		    if(desired_output==m_i)
		    	hits++;
		    else
		    	misses++;
		    printf("%d\t\t%d\t\t\t%d\n",count++,desired_output,m_i);
    	}
    }
    printf("---------------------------------------------------------\n");
    float acc = ((float)(hits)/(hits+misses))*100.0;
    printf("Hits : %d, Misses : %d, Accuracy : %f%%\n", hits,misses,acc);




	return 0;
}
