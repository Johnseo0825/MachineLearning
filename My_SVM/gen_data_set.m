clear, clc;

Num_data_sample = 200;

SNRdB = [-5 0 3 5 ];
SNR = 10.^(SNRdB./10);

Ns = 100;
P_N0 = [1 1 1 1];         % [W]

for i=1:length(SNR)
    
    hx(i) = sqrt(SNR(i).*P_N0(i));
    
        for k=1:Num_data_sample            
            N0 = wgn(Ns,1,P_N0(i),'dBW');            
            y_1 = mean( abs(hx(i) + N0).^2 );
            y_0 = mean( abs(N0).^2 );
            Data_set(k,i) = y_0;
            Data_set(Num_data_sample+k,i) = y_1;
        end
end


plot(Data_set), hold on, grid on;