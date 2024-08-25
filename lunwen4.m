clc
clear
close all
load chaos.mat
A=chaos;
[m,n]=size(A);
%归一化数据
for i=1:m
    for j=1:n
        A1(i,j)=(A(i,j)-min(A(:,j)))/(max(A(:,j))-min(A(:,j))); %归一化
    end
end
a=mean(A1); %求均值
for i=1:length(A1)
    Q(i)=A1(i)-a; %求查询向量
end
for i=1:length(Q)
    for j=1:length(A1)
        a1{i}(j)=A1(j)*Q(i); %求得分函数
    end
end
for i=1:length(a1)
    s=0;
    for j=1:length(a1{i})
        s=s+exp(a1{i}(j));
    end
    a2(i)=s;
end
for i=1:length(a1)
    for j=1:length(a1{i})
        a3{i}(j)=exp(a1{i}(j))/a2(i); %求概率
    end
end
for i=1:length(a3)
    s1=0;
    for j=1:length(a3{i})
        s1=s1+a3{i}(j)*A(j);
    end
    A2(i)=s1; %求注意力得分
end
[A2_max,index]=max(A2);
b=mean(a3{index});
b3=find(a3{index}>b);
if b3(1)==1&&b3(end)==length(A)
    b1=b3;
elseif b3(1)~=1&&b3(end)~=length(A)
       b1(2:end-1)=b3;
       b1(1)=1;
       b1(end)=length(A);
elseif b3(1)==1&&b3(end)~=length(A)
       b1(1:end-1)=b3;
       b1(end)=length(A);
elseif b3(1)~=1&&b3(end)==length(A)
       b1(1)=1;
    for i=1:length(b3)
        b1(i+1)=b3(i);
    end
end
for i=1:length(b1)-1
    p{i}=polyfit(b1(i):b1(i+1),A(b1(i):b1(i+1))',1); %求斜率和截距
end
for i=1:length(b1)-1
    delta(i)=var(A(b1(i):b1(i+1)),1); %求离散度
end
for i=1:length(b1)-1
    T(i)=b1(i+1)-b1(i); %求T
end
for i=1:length(b1)-1
    PL(i,:)=[p{i},delta(i),T(i)]; %求线性模糊信息粒
end
%合并
iter=10; %可调
for l=1:iter
index2=find(T==min(T));
%优化初始分割
theta=0.15; %可调,做实验
h=1;
if index2(1)==1||PL(index2(1),4)==1
    b1(2)=b1(1)+PL(index2(1),4)+PL(index2(1)+1,4);
    h=h+1;
end
if index2(2)==2
       h=h+1;
elseif index2(1)==2
       h=h+1;
end
q=0;
if index2(end)==size(PL,1)||PL(index2(end),4)==1
    b1(end-1)=b1(end-2)+PL(index2(end),4)+PL(index2(end)-1,4);
    q=q+1;
end
for i=h:length(index2)-q
    if abs(PL(index2(i),1)-PL(index2(i)-1,1))<=theta&&(PL(index2(i),1)*PL(index2(i)-1,1))>0
        b1(index2(i))=b1(index2(i)-1)+PL(index2(i),4)+PL(index2(i)-1,4);
        PL(index2(i)-1,1)=theta+PL(index2(i)-1,1)+abs(PL(index2(i)-2,1)-PL(index2(i)-1,1));
        PL(index2(i),1)=theta+PL(index2(i),1)+abs(PL(index2(i)+1,1)-PL(index2(i),1));
    elseif abs(PL(index2(i),1)-PL(index2(i)+1,1))<=theta&&(PL(index2(i),1)*PL(index2(i)+1,1))>0
        b1(index2(i)+1)=b1(index2(i))+PL(index2(i),4)+PL(index2(i)+1,4);
        PL(index2(i)+1,1)=theta+PL(index2(i)+1,1)+abs(PL(index2(i)+2,1)-PL(index2(i)+1,1));
    elseif (PL(index2(i),1)*PL(index2(i)-1,1))<0&&(PL(index2(i),1)*PL(index2(i)+1,1))<0&&PL(index2(i),4)==1
        p1=polyfit(b1(index2(i)-1):b1(index2(i)+PL(index2(i),4)),A(b1(index2(i)-1):b1(index2(i)+PL(index2(i),4)))',1);
        p2=polyfit(b1(index2(i)):b1(index2(i)+1+PL(index2(i)+1,4)),A(b1(index2(i)):b1(index2(i)+1+PL(index2(i)+1,4)))',1);
        LI1=(var(A(b1(index2(i)):b1(index2(i)+1)+PL(index2(i)+1,4)),1))*(sqrt(1/(1+(p2(1)).^2)));
        LI2=(var(A(b1(index2(i)-1):b1(index2(i))+PL(index2(i),4)),1))*(sqrt(1/(1+(p1(1)).^2)));
        if LI1<=LI2
           b1(index2(i)+1)=b1(index2(i))+PL(index2(i),4)+PL(index2(i)+1,4);
           PL(index2(i)+1,1)=theta+PL(index2(i)+1,1)+abs(PL(index2(i)+2,1)-PL(index2(i)+1,1));
        else b1(index2(i))=b1(index2(i)-1)+PL(index2(i),4)+PL(index2(i)-1,4);
             PL(index2(i)-1,1)=theta+PL(index2(i)-1,1)+abs(PL(index2(i)-2,1)-PL(index2(i)-1,1));
             PL(index2(i),1)=theta+PL(index2(i),1)+abs(PL(index2(i)+1,1)-PL(index2(i),1));
        end
    end
end
b1=sort(b1);
for i=1:length(b1)-1
    if b1(i)==b1(i+1)
        b1(i)=0;
    elseif b1(i+1)-b1(i)==1
        b1(i)=0;
    elseif b1(i+1)>length(A)
        b1(i+1)=0;
    end
end
b1(b1==0)=[];
for i=1:length(b1)-1
    if b1(i+1)-b1(i)==1
        b1(i)=0;
    end
end
b1(b1==0)=[];
b2=cell(1,length(b1)-1);
for i=1:length(b1)-1
    b2{i}=polyfit(b1(i):b1(i+1),A(b1(i):b1(i+1))',1); %求斜率和截距
end
delta1=zeros(1,length(b1)-1);
for i=1:length(b1)-1
    delta1(i)=var(A(b1(i):b1(i+1)),1); %求离散度
end
T1=zeros(1,length(b1)-1);
for i=1:length(b1)-1
    T1(i)=b1(i+1)-b1(i); %求T
end
PL1=zeros(length(b1)-1,4);
for i=1:length(b1)-1
    PL1(i,:)=[b2{i},delta1(i),T1(i)]; %求线性模糊信息粒
end
PL=PL1;
T=T1;
end
%分割
for k=1:iter
index1=find(T==max(T));
for i=1:length(index1)
    b4=round(PL(index1(i),4)/2)+1;
    p3=polyfit(b1(index1(i)):b1(index1(i))++b4-1,A(b1(index1(i)):b1(index1(i))++b4-1)',1);
    p4=polyfit(b1(index1(i)+1)-b4+1:b1(index1(i)+1),A(b1(index1(i)+1)-b4+1:b1(index1(i)+1))',1);
    LI3=(var(A(b1(index1(i)):b1(index1(i))++b4-1),1))*(sqrt(1/(1+(p3(1)).^2)));
    LI4=(var(A(b1(index1(i)+1)-b4+1:b1(index1(i)+1)),1))*(sqrt(1/(1+(p4(1)).^2)));
    LI5=(PL(index1(i),3))*(sqrt(1/(1+(PL(index1(i),1)).^2)));
    if LI3<LI5
        b1=[b1,b1(index1(i))+b4-1];
    end
    if LI4<LI5
        b1=[b1,b1(index1(i)+1)-b4+1];
    end
end
b1=sort(b1);
for i=1:length(b1)-1
    if b1(i)==b1(i+1)
        b1(i)=0;
    elseif b1(i+1)-b1(i)==1
        b1(i)=0;
    elseif b1(i+1)>length(A)
        b1(i+1)=0;
    end
end
b1(b1==0)=[];
b2=cell(1,length(b1)-1);
for i=1:length(b1)-1
    b2{i}=polyfit(b1(i):b1(i+1),A(b1(i):b1(i+1))',1); %求斜率和截距
end
delta1=zeros(1,length(b1)-1);
for i=1:length(b1)-1
    delta1(i)=var(A(b1(i):b1(i+1)),1); %求离散度
end
T1=zeros(1,length(b1)-1);
for i=1:length(b1)-1
    T1(i)=b1(i+1)-b1(i); %求T
end
PL1=zeros(length(b1)-1,4);
for i=1:length(b1)-1
    PL1(i,:)=[b2{i},delta1(i),T1(i)]; %求线性模糊信息粒
end
PL=PL1;
T=T1;
end
%聚类
k_max=max(PL(:,1));
k_min=min(PL(:,1));
K=round((k_max-k_min)/theta); %可调
for i=1:size(PL,1)
    LI(i)=(PL(i,3))*(sqrt(1/(1+(PL(i,1)).^2))); %求每个模糊信息粒的线性度指标
end
[LI_min,r]=min(LI);
index3=[];
for h=1:K
    index3=[index3,r]; %确定聚类中心 
    for i=1:size(PL,1)
        for j=1:length(index3)
            T0=min(PL(i,4),PL(index3(j),4));
            HD{i}(j)=(1/2)*abs(PL(i,1)-PL(index3(j),1))*PL(i,4)*PL(index3(j),4)+(sqrt(2*pi)/2)*(abs(PL(i,3)-PL(index3(j),3))*T0+PL(i,3)*(PL(i,4)-T0)+PL(index3(j),3)*(PL(index3(j),4)-T0)); %计算每个样本与当前已有聚类中心之间的距离
        end
        min_HD(i)=min(HD{i}); %计算每个样本与当前已有聚类中心之间的最短距离
    end
    [max_HD,r]=max(min_HD);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
end
for i=1:length(index3)
    b5(i,:)=PL(index3(i),:);
end
iter1=5;
for l1=1:iter1
    for i=1:size(PL,1)
        for j=1:length(index3)
            T0=min(PL(i,4),PL(index3(j),4));
            D1(i,j)=(1/2)*abs(PL(i,1)-PL(index3(j),1))*PL(i,4)*PL(index3(j),4)+(sqrt(2*pi)/2)*(abs(PL(i,3)-PL(index3(j),3))*T0+PL(i,3)*(PL(i,4)-T0)+PL(index3(j),3)*(PL(index3(j),4)-T0)); %计算样本与各聚类中心的距离
        end
    end  
[m1,n1]=size(D1);
C=cell(1,K);
for i=1:m1
    [min_D1,index4]=min(D1(i,:));
    C{index4}= [C{index4},i]; %确定样本的簇标记
end
%计算新的聚类中心
for i=1:length(C)
    c=C{i};
    S1=zeros(1,4);
    for j=1:length(c)
        d=c(j);
        S1=S1+PL(d,:);
        S=S1/length(c);
    end
    b6(i,:)=S;
end
%判断聚类结束条件
for i=1:size(b5,1)
    if b6(i,:)~=b5(i,:)
    b5(i,:)=b6(i,:);
    else b5(i,:)=b6(i,:);
    end
end
end

for i=1:length(C)
    for j=1:length(C{i})
        T0=min(PL(C{i}(j),4),PL(112,4)); %要更改
        D2{i}(j)=(1/2)*abs(PL(C{i}(j),1)-PL(112,1))*PL(C{i}(j),4)*PL(112,4)+(sqrt(2*pi)/2)*(abs(PL(C{i}(j),3)-PL(112,3))*T0+PL(C{i}(j),3)*(PL(C{i}(j),4)-T0)+PL(112,3)*(PL(112,4)-T0));
    end
end
for i=1:length(D2)
    [min_D2,index5]=min(D2{i});
    ind(i)=index5;
end
for i=1:length(C)
    ind1(i)=C{i}(ind(i));
end
for i=1:length(C)
    if length(find(C{i}==112))~=0
        ind2=i;
    end
ind3=ind1(ind2);
for i=1:6  %要更改
    output1(i)=PL(ind3+1,1)*i+PL(ind3+1,2);
end
output1(output1==0)=[];

for i=1:length(C)
    for j=1:length(C{i})
        T0=min(PL(C{i}(j),4),PL(113,4)); %要更改
        D3{i}(j)=(1/2)*abs(PL(C{i}(j),1)-PL(113,1))*PL(C{i}(j),4)*PL(113,4)+(sqrt(2*pi)/2)*(abs(PL(C{i}(j),3)-PL(113,3))*T0+PL(C{i}(j),3)*(PL(C{i}(j),4)-T0)+PL(113,3)*(PL(113,4)-T0));
    end
end
for i=1:length(D3)
    [min_D3,index6]=min(D3{i});
    ind4(i)=index6;
end
for i=1:length(C)
    ind5(i)=C{i}(ind4(i));
end
for i=1:length(C)
    if length(find(C{i}==113))~=0
        ind6=i;
    end
end
ind7=ind5(ind6);
for i=1:6 %要更改
    output2(i)=PL(ind7+1,1)*i+PL(ind7+1,2);
end
output2(output2==0)=[];


for i=1:length(C)
    for j=1:length(C{i})
        T0=min(PL(C{i}(j),4),PL(152,4)); %要更改
        D4{i}(j)=(1/2)*abs(PL(C{i}(j),1)-PL(114,1))*PL(C{i}(j),4)*PL(114,4)+(sqrt(2*pi)/2)*(abs(PL(C{i}(j),3)-PL(114,3))*T0+PL(C{i}(j),3)*(PL(C{i}(j),4)-T0)+PL(114,3)*(PL(114,4)-T0));
    end
end
for i=1:length(D4)
    [min_D4,index7]=min(D4{i});
    ind8(i)=index7;
end
for i=1:length(C)
    ind9(i)=C{i}(ind8(i));
end
for i=1:length(C)
    if length(find(C{i}==114))~=0
        ind10=i;
    end
end
ind11=ind9(ind10);
for i=1:5  %要更改
    output3(i)=PL(ind11+1,1)*i+PL(ind11+1,2);
end
output3(output3==0)=[];


for i=1:length(C)
    for j=1:length(C{i})
        T0=min(PL(C{i}(j),4),PL(115,4)); %要更改
        D5{i}(j)=(1/2)*abs(PL(C{i}(j),1)-PL(115,1))*PL(C{i}(j),4)*PL(115,4)+(sqrt(2*pi)/2)*(abs(PL(C{i}(j),3)-PL(115,3))*T0+PL(C{i}(j),3)*(PL(C{i}(j),4)-T0)+PL(115,3)*(PL(115,4)-T0));
    end
end
for i=1:length(D5)
    [min_D5,index8]=min(D5{i});
    ind12(i)=index8;
end
for i=1:length(C)
    ind13(i)=C{i}(ind12(i));
end
for i=1:length(C)
    if length(find(C{i}==115))~=0
        ind14=i;
    end
end
ind15=ind13(ind14);
for i=6:10  %要更改
    output4(i)=PL(ind15+1,1)*i+PL(ind15+1,2);
end
output4(output4==0)=[];
output=[output3,output4];
actual=A(1191:1200);
error=actual-output';
SSE=sum(error.^2);
MAE=sum(abs(error))/length(actual);
MSE=error'*error/length(actual);
RMSE=MSE^(1/2);
R=[SSE,MAE,MSE,RMSE]


















