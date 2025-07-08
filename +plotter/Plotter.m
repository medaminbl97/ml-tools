classdef Plotter
    % Hilfsklasse zur Visualisierung von logistischer Regression
    % Enthält Methoden zum Zeichnen von Streudiagrammen und Entscheidungsschwellen
    
    methods(Static)
        
        function plotScatter(x1, x2, y)
            % Zeichnet ein Streudiagramm für beliebige binäre Werte in y
            % z.B. y = [-1, 1], y = [3, 7], ...
            
            % Eindeutige Werte in y finden
            classes = unique(y);
            
            if numel(classes) ~= 2
                error('plotScatter unterstützt nur genau 2 Klassen.');
            end
            
            idx1 = find(y == classes(1));
            idx2 = find(y == classes(2));
            
            plot(x1(idx1), x2(idx1), 'r.', 'LineWidth', 1.2, 'DisplayName', string(classes(1)));
            hold on;
            plot(x1(idx2), x2(idx2), 'g.', 'LineWidth', 1.2, 'DisplayName', string(classes(2)));
            
            legend show;
            hold off;
        end
        
        function plotDecisionBoundary(beta, X, y, varargin)
            % Visualisiert die Entscheidungsschwelle der logistischen Regression
            % beta: Parametervektor
            % X: Trainingsdatenmatrix (inkl. Bias)
            % y: Zielvektor
            % Optional: Grad, xmin, xmax, ymin, ymax

            if size(X, 2) <= 3
                % Lineare Grenze
                plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
                plot_y = (-1./beta(3)) .* (beta(2).*plot_x + beta(1));
                plot(plot_x, plot_y, 'LineWidth', 2);
            else
                % Polynomiale Grenze
                Largin = length(varargin);
                if Largin >= 1
                    Grad = varargin{1};
                else
                    Grad = 2;
                end

                if Largin >= 5
                    xmin = varargin{2};
                    xmax = varargin{3};
                    ymin = varargin{4};
                    ymax = varargin{5};
                    u = linspace(xmin, xmax, 100);
                    v = linspace(ymin, ymax, 100);
                else
                    u = linspace(0, 100, 100);
                    v = linspace(0, 100, 100);
                end

                z = zeros(length(u), length(v));
                for i = 1:length(u)
                    for j = 1:length(v)
                        z(i,j) = PolynomMerkmale(u(i), v(j), Grad) * beta;
                    end
                end
                z = z';  % Für contour transponieren
                contour(u, v, z, [0, 0], 'LineWidth', 2);
            end
        end
        
    end
end
