clc
clear
close all
load baseball.mat
A=baseball;
[m,n]=size(A);
for i=1:m
    for j=1:m
        D1(i,j)=sqrt(sum((A(i,1:n-1)-A(j,1:n-1)).^2)); %����ŷ�Ͼ���
    end
end
for i=1:m
    D1(i,i)=max(D1(i,:)); %�ԶԽ���Ԫ�ؽ�����󻯴���
end
for i=1:m
    [D2(i,:),ind1(i,:)]=sort(D1(i,:));  %��ÿ�н������򣬲�ȡ��Ӧ�±�
end
while k<=n-2
    for i=1:m
        B1(i,:)=ind1(i,1:k+1); %�ҳ�ÿ��������k-��������
    end
    w=zeros(m,m);
    for i=1:m
        for j=1:k+1
            w(i,B1(i,j))=exp(-D1(i,B1(i,j))); %�����������������Ȳ���֮���Ȩ�����
        end
    end
    v1=sort(randperm(m,k+1)); %�����ȡk��������Ϊ��ʼ��������
    lmd=2; %�ɵ�
    for i=1:k+1
        v2(i,:)=A(v1(i),1:n-1); %��ʼ����������
    end
    T=20; %�����������ɵ�
    t=1;
    J0=0;
    while t<=T
    %%����������
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
    %��Ŀ�꺯��ֵ
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
    eps=0.1; %�ɵ�
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
        v(i,:)=v3(i,:)./(sum(u(:,i).^2)); %���¾�������
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
    D3(i)=1/(1+sqrt(sum((A(:,i)-A(:,n)).^2))); %����ŷ�Ͼ������ƶ�
end
[S1,ind3]=sort(D3,'descend');
F=ind3(1:k+1); %�����Ӽ�ѡ��
for i=1:length(C1)
    for j=1:length(C1{i})
        for h=1:length(F)
            A1{i}(j,h)=A(C1{i}(j),F(h)); %��ÿ���ض�Ӧ������ֵ
        end
    end
end
for i=1:length(C1)
    for j=1:length(C1{i})
        A2{i}(j)=A(C1{i}(j),n); 
    end
end
for i=1:length(A2)
    X(i)=mean(A2{i}); %��ÿ���������������������Ե�ƽ��ֵ
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
imageInputLayer([k+1 1 1]) %����㣺���������Ĵ�СΪ��Ӧ��ͼ���С
convolution2dLayer([2 1],16,'Padding','same')%16��3*3��С�ľ���ˣ�����Ϊ1���Ա߽粹0
reluLayer %�����
fullyConnectedLayer(100)   %100���ڵ��ȫ���Ӳ�
fullyConnectedLayer(100)   %100���ڵ��ȫ���Ӳ�
fullyConnectedLayer(1)     %1���ڵ��ȫ���Ӳ�
regressionLayer];          %�ع�㣬���ڼ�����ʧֵ
options = trainingOptions('adam', ...%�Ż��㷨������Ӧѧϰ��
'MaxEpochs',100, ...%����������
'MiniBatchSize',128, ...%��С����������
'InitialLearnRate',0.01, ...%��ʼѧϰ��
'GradientThreshold',1, ...%��ֹ�ݶȱ�ը
'Verbose',false,...
'Plots','none','ExecutionEnvironment', 'cpu');%������֤�������ܵ����ݣ�����֤��
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
        
        
                
    
  
        
    
    
    
            
