classdef Plotter
    % Hilfsklasse zur Visualisierung von logistischer Regression
    % Enthält Methoden zum Zeichnen von Streudiagrammen und Entscheidungsschwellen
    
    methods(Static)
        
        function plotScatter(x1, x2, y)
            % Zeichnet ein Streudiagramm der Datenpunkte (x1, x2) mit y ∈ {0,1}
            Einsen = find(y == 1); 
            Nullen = find(y == 0);

            plot(x1(Einsen), x2(Einsen), 'g+','LineWidth', 1.2);
            hold on;
            plot(x1(Nullen), x2(Nullen), 'ro', 'LineWidth', 1.2);
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
