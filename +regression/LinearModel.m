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

        function h = predictSimple(obj, index)
            % Führt fitSimple durch und berechnet h = β₀ + β₁ * x (für Trainingsdaten)
            % Gibt nur den Vorhersagevektor h zurück
    
            % Parameter berechnen
            obj.fitSimple(index);
    
            % Eingangsvariable
            x = obj.X(:, index);
    
            % Vorhersage für Trainingsdaten
            beta0 = obj.B(1);
            beta1 = obj.B(2);
            h = beta0 + beta1 * x;
    
            % Optional im Objekt speichern
            obj.h = h;
        end

        function y_hat = evaluateSimple(obj, x_input, index)
            % Führt fitSimple(index) durch und gibt Vorhersage für x_input zurück
    
            % Parameter berechnen (für aktuelles Feature)
            obj.fitSimple(index);
    
            % Berechne Vorhersage
            beta0 = obj.B(1);
            beta1 = obj.B(2);
            y_hat = beta0 + beta1 * x_input;
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

        function fit(obj)
           % Fit für multivariate lineare Regression (Normalengleichung)
            obj.B = pinv(obj.X) * obj.y;
        end
        
        function C = cost(obj)
            % Kostenfunktion in Matrixschreibweise
            % C = (1/2m) * (y - X*B)' * (y - X*B)
            
            m = size(obj.X, 1);
            e = obj.y - obj.X * obj.B;
            obj.h = obj.X * obj.B;  % Optional: speichern
            C = (1 / (2 * m)) * (e' * e);
        end

        function h = predict(obj)
            % Berechnet Vorhersagevektor für alle Trainingsdaten
            obj.fit();
            h = obj.X * obj.B;
            obj.h = h;
        end

        function y_hat = evaluate(obj, x_input)
            % Gibt Vorhersage für gegebenen Eingabevektor x_input zurück
            % x_input muss ein Zeilenvektor inkl. Bias sein, also z.B. [1, x3, x4, x6]
            obj.fit();
            y_hat = x_input * obj.B;
        end

        function score = r2(obj)
            % Berechnet das Bestimmtheitsmaß R² für das aktuelle Modell
    
            y = obj.y;
            y_mean = mean(y);
    
            % Falls h nicht gesetzt ist, berechne es
            if isempty(obj.h)
                obj.h = obj.X * obj.B;
            end
    
            ss_res = sum((y - obj.h).^2);
            ss_tot = sum((y - y_mean).^2);
    
            score = 1 - (ss_res / ss_tot);
        end
    end
end
