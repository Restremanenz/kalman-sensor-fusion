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

    def set_boundary_condition(self, p, v, q, ba, bg, bm, P_post):
        """Speichert den durch die Boundary Conditions erzwungenen Zielzustand."""
        self.boundary_state = (p.copy(), v.copy(), q, ba.copy(), bg.copy(), bm.copy())
        self.P_post_history[-1] = P_post.copy()

    def smooth(self):
        """Führt den RTS-Rückwärtsdurchlauf durch."""
        print(" -> Wende physikalisch optimalen RTS-Smoother an (Rückwärtsdurchlauf)...")
        N = len(self.states_history) - 1

        smoothed_states = [None] * (N + 1)
        delta_x = np.zeros(self.dim_x)

        # BUGFIX: Berechne den initialen Fehlervektor aus der Boundary-Bedingung!
        if hasattr(self, 'boundary_state'):
            p_fwd, v_fwd, q_fwd, ba_fwd, bg_fwd, bm_fwd = self.states_history[-1]
            p_b, v_b, q_b, ba_b, bg_b, bm_b = self.boundary_state
            
            # Position & Velocity Error
            delta_x[0:3] = p_b - p_fwd
            delta_x[3:6] = v_b - v_fwd
            
            # Rotation Error (q_bound = delta_q * q_fwd)
            error_rot = q_b * q_fwd.inv()
            delta_x[6:9] = error_rot.as_rotvec()
            
            # Bias Errors
            delta_x[9:12] = ba_b - ba_fwd
            delta_x[12:15] = bg_b - bg_fwd
            if self.use_18_state:
                delta_x[15:18] = bm_b - bm_fwd
                
            smoothed_states[N] = self.boundary_state
        else:
            smoothed_states[N] = self.states_history[N]

        for k in range(N - 1, -1, -1):
            P_post = self.P_post_history[k]
            F = self.F_history[k]
            P_prior = self.P_prior_history[k] 

            C_k = P_post @ F.T @ np.linalg.pinv(P_prior)
            
            # Jetzt wird der reale Fehler rückwärts verteilt!
            delta_x = C_k @ delta_x
            
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