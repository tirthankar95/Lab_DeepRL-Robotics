#include<bits/stdc++.h>
using namespace std;
#define MAX_N 4
typedef pair<int,double> ii;
////////////////////////////////////////////
double W=-1,R=50,P=-50;
double pb_p=0.7;
double pb_q;
double discount=0.9;
double reward[MAX_N][MAX_N];
double vk[MAX_N][MAX_N];
double vkk[MAX_N][MAX_N];//temp array.
enum{
    UP,
    RIGHT,
    DOWN,
    LEFT
};
vector<ii> policy[MAX_N][MAX_N];
vector<ii> policyN[MAX_N][MAX_N];
bool cmp(){
    for(int i=0;i<MAX_N;i++){
        for(int j=0;j<MAX_N;j++){
            if((i==0 && j==MAX_N-1)||(i==1 && j==MAX_N-1)||(i==1 && j==1))continue;
            if(policy[i][j].size()!=policyN[i][j].size())return 1;
            for(int k=0;k<policy[i][j].size();k++){
                if(policy[i][j][k].first!=policyN[i][j][k].first || policy[i][j][k].second!=policyN[i][j][k].second)
                    return 1;
            }
        }
    }//
    return 0;
}
////////////////////////////////////////////
void clear_stateValue(){
    memset(vk,0,sizeof(vk));
    memset(vkk,0,sizeof(vkk));
    vkk[0][MAX_N-1]=vk[0][MAX_N-1]=R;
    vkk[1][MAX_N-1]=vk[1][MAX_N-1]=P;
}
void create_grid(){
    pb_q=double(double(1-pb_p))/2;
    memset(vk,0,sizeof(vk));
    memset(vkk,0,sizeof(vkk));
    for(int i=0;i<MAX_N;i++)
    {
        for(int j=0;j<MAX_N;j++){
            reward[i][j]=W;
            policy[i][j].clear();
        }
    }//end of for-i.
    reward[0][MAX_N-1]=reward[1][MAX_N-1]=0;
    vkk[0][MAX_N-1]=vk[0][MAX_N-1]=R;
    vkk[1][MAX_N-1]=vk[1][MAX_N-1]=P;
}
void policy_evaluation(){
    double delta=0.1;
    int iter=0;
    printf("ITER[%d] ->\n",iter);
    for(int i=0;i<MAX_N;i++)
    {
        for(int j=0;j<MAX_N;j++)
        {
            if(i==1 && j==1){
                printf("XX.XX ");
                continue;
            }
            printf("%.2f, ",vk[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    double up[2],right[2],down[2],left[2];// next state <imm. reward,value fn.>
    double temp_inner,temp_outer;
    while(1)
    {
        double max_change=0;
        for(int i=0;i<MAX_N;i++)
        {
            for(int j=0;j<MAX_N;j++){
                if((i==0 && j==MAX_N-1)||(i==1 && j==MAX_N-1)||(i==1 && j==1))continue;
                temp_outer=0;
                up[0]=(i-1<0)?-1:reward[i-1][j];         up[1]=(discount)*((i-1<0)?vk[i][j]:((i-1==1 && j==1)?vk[i][j]:vk[i-1][j]));
                right[0]=(j+1>=MAX_N)?-1:reward[i][j+1]; right[1]=(discount)*((j+1>=MAX_N)?vk[i][j]:((i==1 && j+1==1)?vk[i][j]:vk[i][j+1]));
                down[0]=(i+1>=MAX_N)?-1:reward[i+1][j];  down[1]=(discount)*((i+1>=MAX_N)?vk[i][j]:((i+1==1 && j==1)?vk[i][j]:vk[i+1][j]));
                left[0]=(j-1<0)?-1:reward[i][j-1];       left[1]=(discount)*((j-1<0)?vk[i][j]:((i==1 && j-1==1)?vk[i][j]:vk[i][j-1]));
                for(int k=0;k<policy[i][j].size();k++){
                    switch(policy[i][j][k].first){
                        case UP:
                            temp_inner=pb_p*(up[0]+up[1])\
                                        +pb_q*(right[0]+right[1])\
                                        +pb_q*(left[0]+left[1]);
                            break;
                        case RIGHT:
                            temp_inner=pb_p*(right[0]+right[1])\
                                        +pb_q*(up[0]+up[1])\
                                        +pb_q*(down[0]+down[1]);
                            break;
                        case DOWN:
                            temp_inner=pb_p*(down[0]+down[1])\
                                        +pb_q*(right[0]+right[1])\
                                        +pb_q*(left[0]+left[1]);
                            break;
                        case LEFT:
                            temp_inner=pb_p*(left[0]+left[1])\
                                        +pb_q*(up[0]+up[1])\
                                        +pb_q*(down[0]+down[1]);
                            break;
                    }//end of switch.
                    temp_outer+=policy[i][j][k].second*temp_inner;
                }//end of for-k.
                vkk[i][j]=temp_outer;
                max_change=max(max_change,abs(vkk[i][j]-vk[i][j]));
            }//end of for-j.
        }//end of for-i.
        //COPY vkk into vk;
        iter++;
        printf("ITER[%d] ->\n",iter);
        for(int i=0;i<MAX_N;i++)
        {
            for(int j=0;j<MAX_N;j++)
            {
                if(i==1 && j==1){
                    printf("XX.XX ");
                    continue;
                }
                vk[i][j]=vkk[i][j];
                printf("%.2f ",vk[i][j]);
            }
            printf("\n");
        }
        printf("\n");
        if(max_change<=delta)
            break;
    }
}
void policy_iteration(){
    double cnt,temp_inner,temp_inner_;
    double up[2],right[2],down[2],left[2];// next state <imm. reward,value fn.>
    for(int i=0;i<MAX_N;i++){
        for(int j=0;j<MAX_N;j++){
            if((i==0 && j==MAX_N-1)||(i==1 && j==MAX_N-1)||(i==1 && j==1))continue;
            policyN[i][j].clear();
            up[0]=(i-1<0)?-1:reward[i-1][j];         up[1]=(discount)*((i-1<0)?vk[i][j]:((i-1==1 && j==1)?vk[i][j]:vk[i-1][j]));
            right[0]=(j+1>=MAX_N)?-1:reward[i][j+1]; right[1]=(discount)*((j+1>=MAX_N)?vk[i][j]:((i==1 && j+1==1)?vk[i][j]:vk[i][j+1]));
            down[0]=(i+1>=MAX_N)?-1:reward[i+1][j];  down[1]=(discount)*((i+1>=MAX_N)?vk[i][j]:((i+1==1 && j==1)?vk[i][j]:vk[i+1][j]));
            left[0]=(j-1<0)?-1:reward[i][j-1];       left[1]=(discount)*((j-1<0)?vk[i][j]:((i==1 && j-1==1)?vk[i][j]:vk[i][j-1]));
            vector<int> action;
            switch(UP){
                case UP:
                    temp_inner=pb_p*(up[0]+up[1])\
                                +pb_q*(right[0]+right[1])\
                                +pb_q*(left[0]+left[1]);
                    action.push_back(UP);
                case RIGHT:
                    temp_inner_=pb_p*(right[0]+right[1])\
                                    +pb_q*(up[0]+up[1])\
                                    +pb_q*(down[0]+down[1]);
                    if(temp_inner_>=temp_inner){
                        if(temp_inner_>temp_inner){
                            while(action.size()!=0)action.pop_back();
                        }
                        action.push_back(RIGHT);
                        temp_inner=temp_inner_;
                    }
                    case DOWN:
                        temp_inner_=pb_p*(down[0]+down[1])\
                                    +pb_q*(right[0]+right[1])\
                                    +pb_q*(left[0]+left[1]);
                        if(temp_inner_>=temp_inner){
                            if(temp_inner_>temp_inner){
                                while(action.size()!=0)action.pop_back();
                            }
                            action.push_back(DOWN);
                            temp_inner=temp_inner_;
                        }
                    case LEFT:
                        temp_inner_=pb_p*(left[0]+left[1])\
                                    +pb_q*(up[0]+up[1])\
                                    +pb_q*(down[0]+down[1]);
                        if(temp_inner_>=temp_inner){
                            if(temp_inner_>temp_inner){
                                while(action.size()!=0)action.pop_back();
                            }
                            action.push_back(LEFT);
                            temp_inner=temp_inner_;
                        }
            }//end of switch.
            cnt=((double)1/action.size());
            for(int k=0;k<action.size();k++)
                policyN[i][j].push_back(ii(action[k],cnt));
        }
    }
    for(int i=0;i<MAX_N;i++){
        for(int j=0;j<MAX_N;j++){
            if((i==0 && j==MAX_N-1)||(i==1 && j==MAX_N-1)||(i==1 && j==1)){
                printf("(  )  ");
                continue;
            }
            printf("(");
            for(int k=0;k<policyN[i][j].size();k++){
                switch (policyN[i][j][k].first)
                {
                case UP:
                    printf("U,");
                    break;
                case RIGHT:
                    printf("R,");
                    break;
                case DOWN:
                    printf("D,");
                    break;
                case LEFT:
                    printf("L,");
                    break;                
                }
                
            }
            printf(")  ");
        }
        printf("\n");
    }  
    printf("\n");
}
void update_policy(){
    for(int i=0;i<MAX_N;i++){
        for(int j=0;j<MAX_N;j++){
            policy[i][j]=policyN[i][j];
        }
    }
}
void value_iteration(){
    double delta=0.1;
    int iter=0;
    printf("ITER[%d] ->\n",iter);
    for(int i=0;i<MAX_N;i++)
    {
        for(int j=0;j<MAX_N;j++)
        {
            if(i==1 && j==1){
                printf("XX.XX ");
                continue;
            }
            printf("%.2f, ",vk[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    double up[2],right[2],down[2],left[2];// next state <imm. reward,value fn.>
    double temp_inner;
    while(1)
    {
        double max_change=0;
        for(int i=0;i<MAX_N;i++)
        {
            for(int j=0;j<MAX_N;j++){
                if((i==0 && j==MAX_N-1)||(i==1 && j==MAX_N-1)||(i==1 && j==1))continue;
                up[0]=(i-1<0)?-1:reward[i-1][j];         up[1]=(discount)*((i-1<0)?vk[i][j]:((i-1==1 && j==1)?vk[i][j]:vk[i-1][j]));
                right[0]=(j+1>=MAX_N)?-1:reward[i][j+1]; right[1]=(discount)*((j+1>=MAX_N)?vk[i][j]:((i==1 && j+1==1)?vk[i][j]:vk[i][j+1]));
                down[0]=(i+1>=MAX_N)?-1:reward[i+1][j];  down[1]=(discount)*((i+1>=MAX_N)?vk[i][j]:((i+1==1 && j==1)?vk[i][j]:vk[i+1][j]));
                left[0]=(j-1<0)?-1:reward[i][j-1];       left[1]=(discount)*((j-1<0)?vk[i][j]:((i==1 && j-1==1)?vk[i][j]:vk[i][j-1]));
                switch(UP){
                    case UP:
                        temp_inner=pb_p*(up[0]+up[1])\
                                    +pb_q*(right[0]+right[1])\
                                    +pb_q*(left[0]+left[1]);
                    case RIGHT:
                        temp_inner=max(temp_inner,pb_p*(right[0]+right[1])\
                                    +pb_q*(up[0]+up[1])\
                                    +pb_q*(down[0]+down[1]));
                    case DOWN:
                        temp_inner=max(temp_inner,pb_p*(down[0]+down[1])\
                                    +pb_q*(right[0]+right[1])\
                                    +pb_q*(left[0]+left[1]));
                    case LEFT:
                        temp_inner=max(temp_inner,pb_p*(left[0]+left[1])\
                                    +pb_q*(up[0]+up[1])\
                                    +pb_q*(down[0]+down[1]));
                    }//end of switch.
                vkk[i][j]=temp_inner;
                max_change=max(max_change,abs(vkk[i][j]-vk[i][j]));
            }//end of for-j.
        }//end of for-i.
        //COPY vkk into vk;
        iter++;
        printf("ITER[%d] ->\n",iter);
        for(int i=0;i<MAX_N;i++)
        {
            for(int j=0;j<MAX_N;j++)
            {
                if(i==1 && j==1){
                    printf("XX.XX ");
                    continue;
                }
                vk[i][j]=vkk[i][j];
                printf("%.2f ",vk[i][j]);
            }
            printf("\n");
        }
        printf("\n");
        if(max_change<=delta)
            break;
    }
}
////////////////////////////////////////////
//Driver Function.
int main(){
//1.
    create_grid();
//2.
    //RANDOM POLICY
    double p=(double)1/4;
    for(int i=0;i<MAX_N;i++){
        for(int j=0;j<MAX_N;j++){
            if((i==0 && j==MAX_N-1)||(i==1 && j==MAX_N-1)||(i==1 && j==1))continue;
            for(int k=0;k<4;k++)
                policy[i][j].push_back(ii(k,p));
        }
    }
//3. 
    int pIter=0;
    do{
        pIter+=1;
        printf("POLICY EVALUTION[%d]\n",pIter);
        policy_evaluation();
        printf("POLICY ITERATION[%d]\n",pIter);
        policy_iteration();
        if(!cmp())break;//cmp: 0 means policies are equal.
        update_policy();
        clear_stateValue();
        printf("_____________________________________\n");
    }while(pIter<=10);
    if(pIter>10)printf("DIDN'T CONVERGE.\n\n");
// 4.
    printf("VALUE ITERATION\n");
    clear_stateValue();
    value_iteration();
    policy_iteration();
    return 0;
}