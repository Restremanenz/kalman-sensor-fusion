import numpy as np
from scipy.spatial.transform import Rotation as R

class ESKFSmoother:
    """
    State-of-the-Art Rauch-Tung-Striebel (RTS) Smoother für Error-State Kalman Filter.
    Zeichnet den Vorwärts-Pass auf und glättet die Trajektorie basierend auf der Kovarianz.
    """
    def __init__(self, use_18_state):
        self.use_18_state = use_18_state
        self.dim_x = 18 if use_18_state else 15

        # Speicher für den Vorwärtsdurchlauf
        self.states_history = []      # Nominale Zustände: (p, v, q, ba, bg, bm)
        self.P_post_history = []      # Kovarianz NACH dem Update P_{k|k}
        self.P_prior_history = []     # Kovarianz VOR dem Update P_{k+1|k}
        self.F_history = []           # Error-State Transition Matrix F_k

    def save_initial_state(self, p, v, q, ba, bg, bm, P_post):
        """Speichert den exakten Startzustand VOR der Schleife."""
        self.states_history.append((p.copy(), v.copy(), q, ba.copy(), bg.copy(), bm.copy()))
        self.P_post_history.append(P_post.copy())

    def save_predict(self, F, P_prior):
        """Speichert Parameter direkt nach eskf.predict()"""
        self.F_history.append(F.copy())
        self.P_prior_history.append(P_prior.copy())

    def save_update(self, p, v, q, ba, bg, bm, P_post):
        """Speichert Parameter nach allen asynchronen Updates eines Zeitschritts."""
        self.states_history.append((p.copy(), v.copy(), q, ba.copy(), bg.copy(), bm.copy()))
        self.P_post_history.append(P_post.copy())

    def overwrite_last_update(self, p, v, q, ba, bg, bm, P_post):
        """Überschreibt den letzten Zustand und merkt sich den verursachten Fehler."""
        new_state = (p.copy(), v.copy(), q, ba.copy(), bg.copy(), bm.copy())
        self.boundary_delta_x = self._calculate_error_state(self.states_history[-1], new_state)

        self.states_history[-1] = new_state
        self.P_post_history[-1] = P_post.copy()

    def smooth(self):
        """Führt den RTS-Rückwärtsdurchlauf durch."""
        print(" -> Wende physikalisch optimalen RTS-Smoother an (Rückwärtsdurchlauf)...")
        N = len(self.states_history) - 1

        smoothed_states = [None] * (N + 1)
        smoothed_states[N] = self.states_history[N]

        if hasattr(self, 'boundary_delta_x'):
            delta_x = self.boundary_delta_x.copy()
        else:
            delta_x = np.zeros(self.dim_x)

        for k in range(N - 1, -1, -1):
            P_post = self.P_post_history[k]
            F = self.F_history[k]
            P_prior = self.P_prior_history[k] # Dies ist mathematisch P_{k+1|k}

            # Smoother Gain berechnen: C_k = P_{k|k} * F_k^T * P_{k+1|k}^-1
            C_k = P_post @ F.T @ np.linalg.pinv(P_prior)

            # Rückwärts-Propagation des Error-States
            delta_x = C_k @ delta_x

            # Injiziere den rückwärts berechneten Fehler in den Vorwärts-Zustand
            smoothed_states[k] = self._inject(self.states_history[k], delta_x)

        return self._format_output(smoothed_states)

    def get_forward_states(self):
        """Gibt reine Vorwärts-Zustände zurück (falls Smoother deaktiviert)."""
        return self._format_output(self.states_history)

    def _inject(self, nominal_state, delta_x):
        """Injiziert den Fehlervektor (delta_x) exakt wie dein Kalman-Filter in den Status."""
        p, v, q, ba, bg, bm = nominal_state

        p_new = p + delta_x[0:3]
        v_new = v + delta_x[3:6]

        error_rot = R.from_rotvec(delta_x[6:9])
        q_new = error_rot * q

        ba_new = ba + delta_x[9:12]
        bg_new = bg + delta_x[12:15]

        bm_new = bm.copy()
        if self.use_18_state:
            bm_new = bm + delta_x[15:18]

        return (p_new, v_new, q_new, ba_new, bg_new, bm_new)

    def _format_output(self, states):
        # Wir ignorieren [0] (den Ruhezustand VOR der Schleife), damit die Länge 
        # perfekt zu times_plot und den Visualisierungs-Tools passt.
        positions = np.array([s[0] for s in states[1:]])
        velocities = np.array([s[1] for s in states[1:]])
        orientations = [s[2] for s in states[1:]]
        return positions, velocities, orientations
    
    def _calculate_error_state(self, state_old, state_new):
        """Berechnet den Error-State Vektor (delta_x) zwischen altem und neuem Zustand."""
        p_old, v_old, q_old, ba_old, bg_old, bm_old = state_old
        p_new, v_new, q_new, ba_new, bg_new, bm_new = state_new

        delta_x = np.zeros(self.dim_x)
        
        # Position und Geschwindigkeit
        delta_x[0:3] = p_new - p_old
        delta_x[3:6] = v_new - v_old

        # Orientierung (Quaternion Differenz umgewandelt in Rotationsvektor)
        # q_new = q_err * q_old -> q_err = q_new * q_old.inv()
        error_rot = q_new * q_old.inv()
        delta_x[6:9] = error_rot.as_rotvec()

        # Biases
        delta_x[9:12] = ba_new - ba_old
        delta_x[12:15] = bg_new - bg_old

        if self.use_18_state:
            delta_x[15:18] = bm_new - bm_old

        return delta_x