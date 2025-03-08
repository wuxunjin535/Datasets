clc
clear
close all
load baseball.mat
A=baseball;
[m,n]=size(A);
for i=1:m
    for j=1:m
        D1(i,j)=sqrt(sum((A(i,1:n-1)-A(j,1:n-1)).^2)); %计算欧氏距离
    end
end
for i=1:m
    D1(i,i)=max(D1(i,:)); %对对角线元素进行最大化处理
end
for i=1:m
    [D2(i,:),ind1(i,:)]=sort(D1(i,:));  %对每行进行排序，并取对应下标
end
while k<=n-2
    for i=1:m
        B1(i,:)=ind1(i,1:k+1); %找出每个样本的k-近邻样本
    end
    w=zeros(m,m);
    for i=1:m
        for j=1:k+1
            w(i,B1(i,j))=exp(-D1(i,B1(i,j))); %计算相似样本隶属度差异之间的权衡参数
        end
    end
    v1=sort(randperm(m,k+1)); %随机抽取k个样本作为初始聚类中心
    lmd=2; %可调
    for i=1:k+1
        v2(i,:)=A(v1(i),1:n-1); %初始化聚类中心
    end
    T=20; %迭代次数，可调
    t=1;
    J0=0;
    while t<=T
    %%计算隶属度
    for i=1:m
        for j=1:k+1
            u1(i,j)=2*(sum((A(i,1:n-1)-v2(j,:)).^2)+lmd*sum(w(i,:))); 
        end
    end
    for i=1:m
        for j=1:k+1
            u2(i,j)=lmd*sum(w(i,:)); 
        end
    end
    for i=1:m
        u3(i)=(sum(u2(i,:)./u1(i,:))-1)./(sum(1./u1(i,:))); 
    end
    for i=1:m
        for j=1:k+1
            u(i,j)=(u2(i,j)-u3(i))./(u1(i,j));
        end
    end
    %求目标函数值
    J1=0;
    for i=1:m
        for j=1:m
            for h=1:k+1
                J1=J1+lmd*w(i,B1(j,h))*((u(i,h)-u(B1(j,h),h)^2));
            end
        end
    end
    J2=0;
    for i=1:m
        for j=1:k+1
            J2=J2+(u(i,j)^2)*sum((A(i,1:n-1)-v2(j,:)).^2);
        end
    end
    J=J1+J2;
    eps=0.1; %可调
    if abs(J-J0)<=eps
        break
    elseif abs(J-J0)>eps
        t=t+1;
        J0=J;
    v3=zeros(k+1,n-1);
    for i=1:k+1
        for j=1:m
            v3(i,:)=v3(i,:)+((u(j,i)^2).*A(j,1:n-1));
        end
    end
    for i=1:k+1
        v(i,:)=v3(i,:)./(sum(u(:,i).^2)); %更新聚类中心
    end
    v2=v;
    end
    end
    for i=1:m
        [u_max(i),ind2(i)]=max(u(i,:));
    end
    for i=1:k+1
        C1{i}=find(ind2==i);
    end
    C{k}=C1;
    k=k+1;
end
for i=1:n-1
    D3(i)=1/(1+sqrt(sum((A(:,i)-A(:,n)).^2))); %计算欧氏距离相似度
end
[S1,ind3]=sort(D3,'descend');
F=ind3(1:k+1); %特征子集选择
for i=1:length(C1)
    for j=1:length(C1{i})
        for h=1:length(F)
            A1{i}(j,h)=A(C1{i}(j),F(h)); %求每个簇对应的样本值
        end
    end
end
for i=1:length(C1)
    for j=1:length(C1{i})
        A2{i}(j)=A(C1{i}(j),n); 
    end
end
for i=1:length(A2)
    X(i)=mean(A2{i}); %求每个簇中所有样本决策属性的平均值
end
[m1,n1]=size(A1{1});
m2=round(m1*0.7);
train_x=A1{1}(1:m2,:)';
train_y=A2{1}(1:m2);
test_x=A1{1}(m2+1:m1,:)';
test_y=A2{1}(m2+1:m1);
trainD=reshape(train_x,[k+1 1 1 m2]);
textD=reshape(test_x,[k+1 1 1 m1-m2]);
layers = [
imageInputLayer([k+1 1 1]) %输入层：更改输入层的大小为对应的图像大小
convolution2dLayer([2 1],16,'Padding','same')%16个3*3大小的卷积核，步长为1，对边界补0
reluLayer %激活函数
fullyConnectedLayer(100)   %100个节点的全连接层
fullyConnectedLayer(100)   %100个节点的全连接层
fullyConnectedLayer(1)     %1个节点的全连接层
regressionLayer];          %回归层，用于计算损失值
options = trainingOptions('adam', ...%优化算法，自适应学习率
'MaxEpochs',100, ...%最大迭代次数
'MiniBatchSize',128, ...%最小批处理数量
'InitialLearnRate',0.01, ...%初始学习率
'GradientThreshold',1, ...%防止梯度爆炸
'Verbose',false,...
'Plots','none','ExecutionEnvironment', 'cpu');%用于验证网络性能的数据，即验证集
CNNnet = trainNetwork(trainD,train_y',layers,options);
Y=predict(CNNnet,testD);
 for i=1:n-1
    q{i}=rand(n-1,n-1);
    k{i}=rand(n-1,n-1);
    v{i}=rand(n-1,n-1);
end
for i=1:m
    for j=1:n-1
        Q{j}=A(i,1:n-1)*q{j};
        K{j}=A(i,1:n-1)*k{j};
        V{j}=A(i,1:n-1)*v{j};
    end
end
w0=rand(n-1,1);
for i=1:n-1
    Att(i)=V{i}*w0;
end
for i=1:m
    a1(i,:)=A(i,1:n-1)+Att;
end
w1=rand(1,n-1);
for i=1:m
    y(i)=a1(i,:)*w1';
end
error=A(:,end)-y';
MAE=sum(abs(error))/m;
MSE=error'*error/m;
RMSE=MSE^(1/2);
R=[MAE,MSE,RMSE]       
        
        
                
    
  
        
    
    
    
            
