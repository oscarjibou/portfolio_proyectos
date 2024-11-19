% Signal Delay
t = 0:0.1:1; % Time vector
x = 2*cos(2*pi.*t) % Signal

m = 5; % Delay:

y = [zeros(1,m) x(1:end-m)] % Delayed signal:


% Visual comparison of the original and delayed signals:
plot(t,x)
hold on
plot(t,y)
legend('Original','Delayed')

hold off

% Delay in segments 

function [bloque_ret, new_state] = retardador_bloques(bloque, m, state)
    % Programa la función asumiendo que "bloque" y "state" son vectores columna
    % Los vectores de salida "bloque_ret" y "new_state" también deben ser vectores columna

    %Inicializar state con m ceros si state==[] (está vacío). Mira la ayuda de la función isempty
    if isempty(state)
        state = zeros(m,1);
    end


    %Generar salida bloque_ret a partir de state y bloque

    bloque_ret = [state; bloque(1:end-m)];

    %Generar actualizacion del estado new_state para los siguientes bloques
    new_state = bloque(end-m+1:end);

end

% Average power of a signal

function [med, pot] = media_potencia(x)
 
    % Calcula la media de x, med
    
    med = mean(x);
    
    % Calcula la potencia de x, pot
    
    pot = mean(x.^2);
    
    end