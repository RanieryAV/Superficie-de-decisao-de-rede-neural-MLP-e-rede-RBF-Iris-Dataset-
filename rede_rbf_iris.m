%2º Trabalho - Inteligência Computacional - Raniery Alves Vasconcelos - 473532

%2. Implemente uma rede RBF para classificar as amostras do conjunto de vetores
%"iris_log”. A estratégia de validação é a k-fold (k = 5). O programa deve exibir a
%acurácia média das cinco rodadas treinamento/teste.

% Limpar o console sempre que executa o código
clear; close all; clc;
base_original_iris=readmatrix('iris_log.dat');

vetor_de_amostras=base_original_iris(:,1:4)';%Valores/amostras de entrada
vetor_de_rotulos=base_original_iris(:,5:7)';%Rótulos das entradas

vetor_de_amostras
vetor_de_rotulos

numero_de_neuronios=10;
sigma_spread=1;%Máxima distância entre dois centros quaisquer divido pela raiz quadrada do número de centros (estes últimos obtidos por clustering)

[p, N]=size(vetor_de_amostras);%Onde "p" é o número de linhas, e "N" o número de colunas da matriz
fprintf("\nvalor de p = \n"+p);
fprintf("\nvalor de N = \n"+N);

%Normalizar os dados de input
for i=1:p
    vetor_de_amostras(i,:)=(vetor_de_amostras(i,:)-mean(vetor_de_amostras(i,:)))/std(vetor_de_amostras(i,:));
end

fprintf("\nVetor de amostras ANTES de randomizar:\n")
vetor_de_amostras

%Obter indices randomizados
indices_randomizados=randperm(p);
indices_randomizados

%Colocar os valores nos indices sorteados
vetor_de_amostras_aux=vetor_de_amostras;
vetor_de_rotulos_aux=vetor_de_rotulos;
for i=1:p
    vetor_de_amostras(:,i)=vetor_de_amostras_aux(:,(indices_randomizados(i)));
    vetor_de_rotulos(:,i)=vetor_de_rotulos_aux(:,(indices_randomizados(i)));
end

%Mostrar vetor randomizado de amostras
fprintf("\nVetor de amostras DEPOIS de randomizar:\n")
vetor_de_amostras

Z=zeros(numero_de_neuronios,N-30);

contador_acertos=0;

for k=0:p %dobra/fold (indo de 0 até 4, ou seja, 0 -> 1 -> 2 -> 3 -> 4 = 5-fold)
    C=randn(p,numero_de_neuronios);%Criar matriz de números aleatórios distribuídos de forma normal com o mesmo número de linhas de "vetor_de_amostras" e mesmo número de colunas de "numero_de_neuronios"
    %z_teste=zeros(numero_de_neuronios,N-120)

    vetor_de_amostras_aux_para_corte=vetor_de_amostras;
    vetor_de_rotulos_aux_para_corte=vetor_de_rotulos;

    comeco_do_corte_do_vetor=(k*30+1);
    fim_do_corte_do_vetor=(k*30+30);

    contador_for_corte=1;
    for indice=comeco_do_corte_do_vetor:fim_do_corte_do_vetor
        vetor_amostras_teste(:,contador_for_corte)=vetor_de_amostras_aux_para_corte(:,indice);
        vetor_rotulos_teste(:,contador_for_corte)=vetor_de_rotulos_aux_para_corte(:,indice);
        contador_for_corte=contador_for_corte+1;
    end

    vetor_amostras_treino=vetor_de_amostras;
    vetor_amostras_treino(:,comeco_do_corte_do_vetor:fim_do_corte_do_vetor)=[];
    vetor_rotulos_treino=vetor_de_rotulos;
    vetor_rotulos_treino(:,comeco_do_corte_do_vetor:fim_do_corte_do_vetor)=[];
    
    fprintf("\n\nApós entrar no laço, essa é a iteração: "+(k))
    vetor_amostras_treino
    vetor_rotulos_treino
    


    fprintf("\n\nComeco_do_corte_do_vetor: "+(comeco_do_corte_do_vetor))
    fprintf("\n\nFim_do_corte_do_vetor: "+(fim_do_corte_do_vetor))
    %Aqui, "apagar" uma linha do vetor (seja das amostras ou dos rótulos, 
    %referente ao "fold" atual reservado para teste) com elementos consiste
    %em igualar tal linha INTEIRA a zeros (0s)
    
    
    fprintf("\n\nApós deletar, essa é a iteração: "+(k))
    vetor_amostras_treino
    vetor_rotulos_treino

    for i=1:N-30
        for j=1:numero_de_neuronios%Percorrer matriz de amostras para aplicar a função de ativação em cada valor i,j
            u=norm(vetor_amostras_treino(:,i)-C(:,j));
            fu=exp(-u^2/(2*sigma_spread^2));
            Z(j,i)=fu;%Matriz após a aplicação da função de ativação
        end
    end
    a=1;
    Z=[((-1)*ones(1,numero_de_neuronios-30));Z];%Adicionar o (-1) do bias sobre os elementos da matriz
    a=11;
    %Matriz de pesos (aplicação da pseudo-transposta para emular a divisão de matrizes
    M=vetor_rotulos_treino*Z'*(Z*Z')^(-1);%Pesos referentes a camada de saída

    %Código da previsão
    col=1;
    c=1;
    for l=1:(numero_de_neuronios-1)
        u=norm(vetor_amostras_teste-C(:,l));
        fu=exp(-u^2/(2*sigma_spread^2));
        z_teste(l)=fu;%Matriz após a aplicação da função de ativação
    end
    a=1;
    if k>0
        z_teste(numero_de_neuronios)=[];
    end
    z_teste=[-1 z_teste];%Adicionar o (-1) do bias
    z_teste_aux=z_teste';
    valores_preditos=M*z_teste_aux;

    [a, b] = max(valores_preditos);%Encontrar o maior valor na matriz de valores preditos e seu indice
    [c, d] = max(vetor_rotulos_teste);%Encontrar o maior valor na matriz de rótulos dos valores de teste usados na predição e seu indice

    if b==d%Se o indice do maior valor predito é igual ao indice do maior rótulo (1 > 0) DAQUELA amostra, então...
        contador_acertos = contador_acertos + 1;
    end
    valor_acuracia=contador_acertos/(p+1)
end

valor_acuracia=contador_acertos/(p+1)

fprintf("\n\nValor da acurácia média das cinco rodadas treinamento/teste: "+valor_acuracia+"\n");