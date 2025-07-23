classdef KMeansModel < handle
    properties
        X              % Datenmatrix (m x n)
        K              % Anzahl Cluster
        maxIter        % Maximale Iterationen
        tol = 1e-5     % Toleranz für Abbruch (Default 1e-5)
        centroids      % Clusterzentren (mu)
        labels         % Clusterzuordnungen (C)
        cost           % Kostenfunktion J
    end
    
    methods
        function obj = KMeansModel(X, K, maxIter, tol)
            % Konstruktor
            obj.X = X;
            obj.K = K;
            if nargin < 3 || isempty(maxIter), maxIter = 100; end
            if nargin < 4 || isempty(tol), tol = 1e-5; end
            obj.maxIter = maxIter;
            obj.tol = tol;
        end
        
        function fit(obj)
            [m, n] = size(obj.X);
            
            % Initialisierung: Zufällige Auswahl von K Datenpunkten
            rng('shuffle');
            perm = randperm(m);
            mu = obj.X(perm(1:obj.K), :);
            
            J = inf;
            obj.labels = zeros(m, 1);
            
            for iter = 1:obj.maxIter
                % Schritt 1: Zuweisung
                D = pdist2(obj.X, mu);
                [~, C] = min(D, [], 2);
                
                % Schritt 2: Neue Zentren berechnen
                for k = 1:obj.K
                    if any(C == k)
                        mu(k, :) = mean(obj.X(C == k, :), 1);
                    else
                        mu(k, :) = obj.X(randi(m), :);
                    end
                end
                
                % Schritt 3: Kostenfunktion
                J_new = 0;
                for i = 1:m
                    J_new = J_new + sum((obj.X(i,:) - mu(C(i), :)).^2);
                end
                
                if abs(J - J_new) < obj.tol
                    break;
                end
                J = J_new;
            end
            
            obj.centroids = mu;
            obj.labels = C;
            obj.cost = J / m; % Durchschnittlicher Fehler
        end
        
        function plotClusters(obj)
            figure; hold on;
            colors = lines(obj.K);
            for k = 1:obj.K
                scatter(obj.X(obj.labels == k, 1), obj.X(obj.labels == k, 2), 36, colors(k,:), 'filled');
            end
            plot(obj.centroids(:,1), obj.centroids(:,2), 'kx', 'MarkerSize', 12, 'LineWidth', 2);
            title(['K-Means Clustering (K = ', num2str(obj.K), ')']);
            xlabel('x1'); ylabel('x2');
            axis equal;
            grid on;
            hold off;
        end
    end
    
    methods (Static)
        function [bestK, J_all] = elbow(X, k_range, n_runs)
            % Elbow-Methode: gibt bestK und J_all zurück
            if nargin < 3
                n_runs = 20;
            end
            J_all = zeros(length(k_range), 1);
    
            for i = 1:length(k_range)
                K = k_range(i);
                bestJ = inf;
                for run = 1:n_runs
                    model = kmeans.KMeansModel(X, K);
                    model.fit();
                    if model.cost < bestJ
                        bestJ = model.cost;
                    end
                end
                J_all(i) = bestJ;
            end
    
            % Plot
            figure;
            plot(k_range, J_all, '-o', 'LineWidth', 1.5);
            xlabel('Anzahl Cluster K');
            ylabel('Beste Kostenfunktion J');
            title('Elbow-Methode');
            grid on;
    
            % Gerade von erstem zu letztem Punkt
            x1 = k_range(1);
            y1 = J_all(1);
            x2 = k_range(end);
            y2 = J_all(end);
            
            % Richtungsvektor
            lineVec = [x2 - x1, y2 - y1];
            lineVec = lineVec / norm(lineVec);
            
            % Berechnen aller Abstände
            distances = zeros(length(k_range), 1);
            for i = 1:length(k_range)
                p = [k_range(i), J_all(i)];
                a = [x1, y1];
                vec = p - a;
                proj = dot(vec, lineVec) * lineVec;
                orth = vec - proj;
                distances(i) = norm(orth);
            end
            
            % Index mit maximalem Abstand (größter "Knick")
            [~, idx] = max(distances);
            bestK = k_range(idx);
    
            % Hinweistext ausgeben
            disp(['Empfohlene Clusterzahl (Elbow): K = ', num2str(bestK)]);
        end
    end

end
