import numpy as np
from scipy.spatial.transform import Rotation as R


class ESKFSmoother:
    """Rauch-Tung-Striebel-Smoother für einen 15- oder 18-State-ESKF."""

    def __init__(self, use_18_state):
        self.use_18_state = use_18_state
        self.dim_x = 18 if use_18_state else 15

        # Gefilterte Zustände nach allen Messupdates.
        self.states_history = []
        self.P_post_history = []

        # Vorhergesagte Zustände vor den Messupdates.
        self.predicted_states_history = []
        self.P_prior_history = []
        self.F_history = []

        # Wird nach smooth() für Validierungsplots bereitgestellt.
        self.P_smoothed_history = []

    def save_initial_state(self, p, v, q, ba, bg, bm, P_post):
        """Speichert den gefilterten Zustand am Trajektorienstart."""
        self.states_history.append(self._make_state(p, v, q, ba, bg, bm))
        self.P_post_history.append(self._symmetrize(P_post))

    def save_predict(self, p, v, q, ba, bg, bm, F, P_prior):
        """Speichert die Prädiktion vor den Messupdates des Zeitschritts."""
        self.predicted_states_history.append(self._make_state(p, v, q, ba, bg, bm))
        self.F_history.append(F.copy())
        self.P_prior_history.append(self._symmetrize(P_prior))

    def save_update(self, p, v, q, ba, bg, bm, P_post):
        """Speichert den gefilterten Zustand nach allen Messupdates."""
        self.states_history.append(self._make_state(p, v, q, ba, bg, bm))
        self.P_post_history.append(self._symmetrize(P_post))

    def overwrite_last_update(self, p, v, q, ba, bg, bm, P_post):
        """Ersetzt den letzten Zustand durch den Zustand nach den Randbedingungen."""
        if not self.states_history:
            raise RuntimeError("Es ist kein Zustand für die Endbedingung vorhanden.")

        self.states_history[-1] = self._make_state(p, v, q, ba, bg, bm)
        self.P_post_history[-1] = self._symmetrize(P_post)

    def smooth(self):
        """Führt den vollständigen ESKF-RTS-Rückwärtsdurchlauf aus."""
        transition_count = len(self.states_history) - 1

        if transition_count <= 0:
            self.P_smoothed_history = [P.copy() for P in self.P_post_history]
            return self._format_output(self.states_history)

        self._validate_history(transition_count)
        print(" -> Wende vollständigen ESKF-RTS-Smoother an...")

        smoothed_states = [self._copy_state(state) for state in self.states_history]
        smoothed_covariances = [P.copy() for P in self.P_post_history]

        for k in range(transition_count - 1, -1, -1):
            P_post = self.P_post_history[k]
            F_k = self.F_history[k]
            P_prior_next = self.P_prior_history[k]
            cross_covariance = P_post @ F_k.T

            # C_k = P_k+ F_k^T (P_(k+1)-)^-1
            try:
                smoother_gain = np.linalg.solve(
                    P_prior_next.T,
                    cross_covariance.T
                ).T
            except np.linalg.LinAlgError:
                smoother_gain = cross_covariance @ np.linalg.pinv(P_prior_next)

            predicted_next = self.predicted_states_history[k]
            smoothed_next = smoothed_states[k + 1]
            delta_next = self._calculate_error_state(predicted_next, smoothed_next)
            delta_current = smoother_gain @ delta_next
            smoothed_states[k] = self._inject(self.states_history[k], delta_current)

            covariance_correction = smoothed_covariances[k + 1] - P_prior_next
            P_smoothed = (
                P_post
                + smoother_gain @ covariance_correction @ smoother_gain.T
            )
            smoothed_covariances[k] = self._symmetrize(P_smoothed)

        self.P_smoothed_history = smoothed_covariances
        return self._format_output(smoothed_states)

    def get_forward_states(self):
        """Gibt die ungeglätteten Vorwärtszustände zurück."""
        self.P_smoothed_history = [P.copy() for P in self.P_post_history]
        return self._format_output(self.states_history)

    def get_smoothed_covariances(self):
        """Gibt die Kovarianzen passend zur ausgegebenen Trajektorie zurück."""
        if not self.P_smoothed_history:
            return np.empty((0, self.dim_x, self.dim_x))
        return np.asarray(self.P_smoothed_history[1:])

    def _validate_history(self, transition_count):
        """Prüft die Konsistenz der gespeicherten Vorwärts-Historien."""
        if len(self.P_post_history) != len(self.states_history):
            raise RuntimeError(
                "Zustands- und Posteriorhistorie besitzen unterschiedliche Längen."
            )

        required_histories = {
            "vorhergesagte Zustände": len(self.predicted_states_history),
            "Transitionsmatrizen": len(self.F_history),
            "Prior-Kovarianzen": len(self.P_prior_history)
        }
        for name, length in required_histories.items():
            if length < transition_count:
                raise RuntimeError(
                    f"RTS-Historie unvollständig: {name} enthält {length}, "
                    f"benötigt werden {transition_count} Einträge."
                )

    def _inject(self, nominal_state, delta_x):
        """Injiziert einen globalen Error-State in den Nominalzustand."""
        p, v, q, ba, bg, bm = nominal_state

        p_new = p + delta_x[0:3]
        v_new = v + delta_x[3:6]
        q_new = R.from_rotvec(delta_x[6:9]) * q
        ba_new = ba + delta_x[9:12]
        bg_new = bg + delta_x[12:15]
        bm_new = bm + delta_x[15:18] if self.use_18_state else bm.copy()

        return self._make_state(p_new, v_new, q_new, ba_new, bg_new, bm_new)

    def _calculate_error_state(self, reference_state, target_state):
        """Drückt target minus reference im ESKF-Fehlerkoordinatensystem aus."""
        p_ref, v_ref, q_ref, ba_ref, bg_ref, bm_ref = reference_state
        p_target, v_target, q_target, ba_target, bg_target, bm_target = target_state

        delta_x = np.zeros(self.dim_x)
        delta_x[0:3] = p_target - p_ref
        delta_x[3:6] = v_target - v_ref
        delta_x[6:9] = (q_target * q_ref.inv()).as_rotvec()
        delta_x[9:12] = ba_target - ba_ref
        delta_x[12:15] = bg_target - bg_ref

        if self.use_18_state:
            delta_x[15:18] = bm_target - bm_ref

        return delta_x

    @staticmethod
    def _make_state(p, v, q, ba, bg, bm):
        """Erzeugt eine unabhängige Kopie eines Nominalzustands."""
        return (
            np.asarray(p, dtype=float).copy(),
            np.asarray(v, dtype=float).copy(),
            R.from_quat(q.as_quat()),
            np.asarray(ba, dtype=float).copy(),
            np.asarray(bg, dtype=float).copy(),
            np.asarray(bm, dtype=float).copy()
        )

    @classmethod
    def _copy_state(cls, state):
        return cls._make_state(*state)

    @staticmethod
    def _symmetrize(covariance):
        covariance = np.asarray(covariance, dtype=float)
        return 0.5 * (covariance + covariance.T)

    @staticmethod
    def _format_output(states):
        """Entfernt den initialen Zustand passend zur times_plot-Zeitachse."""
        output_states = states[1:]
        if not output_states:
            return np.empty((0, 3)), np.empty((0, 3)), []

        positions = np.asarray([state[0] for state in output_states])
        velocities = np.asarray([state[1] for state in output_states])
        orientations = [state[2] for state in output_states]
        return positions, velocities, orientations
