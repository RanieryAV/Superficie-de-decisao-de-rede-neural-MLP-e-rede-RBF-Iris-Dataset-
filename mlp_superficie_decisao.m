%2º Trabalho - Inteligência Computacional - Raniery Alves Vasconcelos - 473532

%1. Usando o conjunto de dados 2-D disponível no arquivo two_classes.dat, trace a
%superfície de decisão obtida com uma rede neural MLP treinadas com todas as amostras.

base_dados_two_classes=readmatrix("two_classes.dat");

amostras_x_y=base_dados_two_classes(:,1:2);
classes_das_amostras=base_dados_two_classes(:,3);

%Plotar a base de dados para observar a disposição inicial das amostras
gscatter(base_dados_two_classes(:,1),base_dados_two_classes(:,2),classes_das_amostras,'rgb');



rede_mlp=feedforwardnet([5,10]);
rede_mlp.trainFcn='traingdx';

[rede_mlp, registros_de_treino]=train(rede_mlp,amostras_x_y',classes_das_amostras');

registros_de_treino

x1range = min(amostras_x_y(:,1)):.1:max(amostras_x_y(:,1));
x2range = min(amostras_x_y(:,2)):.1:max(amostras_x_y(:,2));
[xx1, xx2] = meshgrid(x1range,x2range);
XGrid = [xx1(:) xx2(:)];

%Analisar quais são os rótulos/targets de saída pós-treino (com os mesmos dados do treino)
saidaMLP = sim(rede_mlp,XGrid');

classes_preditas = saidaMLP';



figure;
hold on;
%subplot(2,2,1);


for indice=1:size(classes_preditas)
    if classes_preditas(indice,:)>0 && classes_preditas(indice,:)<0.6
        classes_preditas(indice,:)=0;
    end
    if classes_preditas(indice,:)<0 && classes_preditas(indice,:)>-1.6
        classes_preditas(indice,:)=0;
    end
end

gscatter(xx1(:), xx2(:), classes_preditas,'rgb');

title("rede mlp (rode o código várias vezes para ver variações da superfície de decisão)")
legend off, axis tight

legend({"Além da superficie de decisão (Classe '-1')","Classe '1' (região rodeada de pontos iguais)"},'Location',[0.35,0.01,0.35,0.05],'Orientation','Horizontal')
%Plotar as regiões de cada classe
%figure;
%hold on;

%x = linspace(0,1000);
%y = linspace(0,1000);
%grid_total = meshgrid(x,y);


%grid_plot=zeros(1000,2);
%plot(amostras_x_y(:,1),amostras_x_y(:,2),"o");

%for indice=1:size(saidaMLP)
%    if saidaMLP(indice)>0.0 && saidaMLP(indice)<1.0
%        grid_plot(indice,:)=1;
%    end
%end
%plot(grid_plot(indice,1),grid_plot(indice,2),"x");