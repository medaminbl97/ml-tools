classdef LogisticModel
    %LOGISTICMODEL Summary of this class goes here
    %   Detailed explanation goes here

    properties
        X   % Eingabematrix inkl. Bias-Term (m × (n+1))
        y   % Zielvektor (m × 1), enthält Zahlen für Klassen (z.B. 1, 2, 3, ...)
        B   % Gewichtsmatrix (n+1 × K), Spalte k = Parameter für Klasse k
        H   % Hypothesenmatrix (m × K)
    end

    methods
        function obj = LogisticModel(X, y)
            if nargin < 2
                error('Bitte X und y beim Erstellen des Modells angeben.');
            end
        
            % Bias-Term hinzufügen
            m = size(X, 1);
            X_aug = [ones(m, 1), X];
        
            % Zielvariable vorbereiten
            y = y(:);  % sicherstellen, dass y Spaltenvektor ist
        
            % Anzahl Klassen bestimmen
            classLabels = unique(y);
            numClasses = numel(classLabels);
        
            % Initialisierung der Parameter
            B = zeros(size(X_aug, 2), numClasses);  % Spaltenweise: β_k für jede Klasse k
        
            % Objektattribute setzen
            obj.X = X_aug;
            obj.y = y;
            obj.B = B;
            obj.H = [];  % Hypothesenmatrix (m × K) wird später berechnet
        end

        function binaryModel = getBinaryModel(obj, value1, value2)
            % Erzeugt ein binäres Modell aus dem aktuellen Multiklassenmodell
            % Nur Daten mit Labels value1 oder value2 werden verwendet
            % y wird in {0,1} umcodiert:  y = 1 <=> value2, y = 0 <=> value1
        
            % Auswahl der Daten mit den gewünschten Klassen
            idx = obj.y == value1 | obj.y == value2;
            X_bin = obj.X(idx, :);
            y_raw = obj.y(idx);
        
            % Neue binäre Zielvariable
            y_bin = double(y_raw == value2);  % value2 wird zur Klasse 1
        
            % Neues binäres Modell erzeugen
            binaryModel = LogisticModel(X_bin(:,2:end), y_bin);  % Bias wird im Konstruktor erneut hinzugefügt
        end
    end
end