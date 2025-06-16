classdef LinearModel < handle
    %LINEARMODEL Klasse für Lineare Regression (ein- oder mehrdimensional)
    
    properties
        X           % Trainingsdaten (m x n)
        y           % Reale Zielwerte (m x 1)
        B           % Parametervektor (Beta)
        h           % Vorhersagewerte (Hypothese)
    end
    
    methods
        function obj = LinearModel(X, y, B)
            % Konstruktor
            obj.X = X;
            obj.y = y;
            obj.B = B;
            obj.h = [];  % wird bei Bedarf berechnet
        end

        function fitSimple(obj,index)
            % Fit für einfache lineare Regression mit einer Eingangsvariable
    
            x = obj.X(:,index);
            y = obj.y;
    
            % Mittelwerte
            x_bar = mean(x);
            y_bar = mean(y);
    
            % Kovarianz und Varianz (cov gibt Matrix zurück)
            cov_xy = cov(x, y);     % 2x2 Matrix
            beta1 = cov_xy(1, 2) / var(x);  % Alternativ: cov_xy(1,2) / cov_xy(1,1)
    
            beta0 = y_bar - beta1 * x_bar;
    
            % Speichern als Spaltenvektor
            obj.B = [beta0; beta1];
        end

        function [lx, ly] = predictSimple(obj, index)
            % Führt fitSimple durch und gibt Regressionslinie zurück
            % index: Spalte von X (ab 2 wegen Bias)
    
            % Parameterschätzung (aktualisiert obj.B)
            obj.fitSimple(index);
    
            % Eingangsvariable extrahieren
            x = obj.X(:, index);
    
            % linspace für glatte Linie
            lx = linspace(min(x), max(x), 100)';
    
            % Vorhersage mit aktuellen Parametern
            beta0 = obj.B(1);
            beta1 = obj.B(2);
            ly = beta0 + beta1 * lx;
        end

        function C = costSimple(obj, index)
            % Berechnet Kostenfunktion für einfache lineare Regression (1 Feature)
            x = obj.X(:, index);
            y = obj.y;
            m = length(y);
    
            % Vorhersage
            beta0 = obj.B(1);
            beta1 = obj.B(2);
            h = beta0 + beta1 * x;
    
            % Speichern optional
            obj.h = h;
    
            % Kostenfunktion (vektorisiert)
            C = (1 / (2 * m)) * sum((h - y).^2);
        end
        
        function C = cost(obj)
            % Kostenfunktion in Matrixschreibweise
            % C = (1/2m) * (y - X*B)' * (y - X*B)
            
            m = size(obj.X, 1);
            e = obj.y - obj.X * obj.B;
            obj.h = obj.X * obj.B;  % Optional: speichern
            C = (1 / (2 * m)) * (e' * e);
        end

    end
end
