
class CustomEnvMcwa(gym.Env):

    # initialization
    def __init__(self, n_cells, wavelength, desired_angle):
        super().__init__()
        # self.eng = matlab.engine.start_matlab()
        self.eng = octave
        # self.eng.addpath(self.eng.genpath(r'RETICOLOLOCATION'));
        self.eng.addpath(self.eng.genpath('solvers'));
        self.eng.addpath(self.eng.genpath('/Users/yongha/project/metasurface/reticolo/V9'))
        os.makedirs('data', exist_ok=True)
        self.eff_file_path = 'data/' + str(wavelength) + '_' + str(desired_angle) + '_' + str(
            n_cells) + '_eff_table.json'
        if Path(self.eff_file_path).exists():
            with open(self.eff_file_path, 'rb') as f:
                self.eff_table = json.load(f)
        else:
            self.eff_table = {}
        self.n_cells = n_cells
        # self.wavelength = matlab.double([wavelength])
        self.wavelength = octave.double([wavelength])

        # self.desired_angle = matlab.double([desired_angle])
        self.desired_angle = octave.double([desired_angle])
        self.struct = np.ones(self.n_cells)
        self.eff = 0

        self.comp_table = pd.DataFrame(
            columns=['reti-1', 'reti0', 'reti+1', 'mcwat-1', 'mcwat0', 'mcwat+1',
                              'mcwas-1', 'mcwas0', 'mcwas+1', 't_reti', 'tpr_reti', 't_mcwat', 'tpr_mcwat', 't_mcwas', 'tpr_mcwas'])
        self.img_table = pd.DataFrame()

    def getEffofStructure(self, struct, wavelength, desired_angle):
        img = struct
        angle = desired_angle
        mcwa = JLABCode()
        eff, _ = mcwa.reproduce_acs(img, wavelength, angle)

        return eff

    def getEffofStructure_benchmark(self, struct, wavelength, desired_angle):
        img = struct
        angle = desired_angle

        t0 = np.array([time.time(), time.process_time()])
        eff, effs_reti = self.eng.Eval_Eff_1D(img, wavelength, angle, nout=2)
        t1 = np.array([time.time(), time.process_time()])
        t_reti = t1 - t0

        mcwa = JLABCode()
        t0 = np.array([time.time(), time.process_time()])
        _, effs_mcwa_tmm = mcwa.reproduce_acs(img, wavelength, angle, algo='TMM')  # +60 deg, 0 deg, - 60 deg
        t1 = np.array([time.time(), time.process_time()])
        t_mcwa_tmm = t1 - t0

        t0 = np.array([time.time(), time.process_time()])
        _, effs_mcwa_smm = mcwa.reproduce_acs(img, wavelength, angle, algo='SMM')  # +60 deg, 0 deg, - 60 deg
        t1 = np.array([time.time(), time.process_time()])
        t_mcwa_smm = t1 - t0

        effs_mcwa_tmm = effs_mcwa_tmm[::-1]  # -60 deg, 0 deg, + 60 deg
        effs_mcwa_smm = effs_mcwa_smm[::-1]  # -60 deg, 0 deg, + 60 deg
        to_add = np.concatenate((effs_reti.flatten(), effs_mcwa_tmm, effs_mcwa_smm, t_reti, t_mcwa_tmm, t_mcwa_smm)).reshape(1, -1)
        to_add = pd.DataFrame(to_add,
                              columns=['reti-1', 'reti0', 'reti+1', 'mcwat-1', 'mcwat0', 'mcwat+1',
                              'mcwas-1', 'mcwas0', 'mcwas+1', 't_reti', 'tpr_reti', 't_mcwat', 'tpr_mcwat', 't_mcwas', 'tpr_mcwas'])

        to_add_img = pd.DataFrame(img)

        self.comp_table = pd.concat([self.comp_table, to_add], ignore_index=True)
        self.img_table = pd.concat([self.img_table, to_add_img], ignore_index=True)

        return eff

    def step(self, action):  # array: input vector, ndarray
        done = False
        result_before = self.eff
        struct_after = self.struct.copy()
        if action == self.n_cells:
            done = True
        elif struct_after[action] == 1:
            struct_after[action] = -1
        elif struct_after[action] == -1:
            struct_after[action] = 1
        else:
            raise ValueError('struct component should be 1 or -1')
        key = tuple(struct_after.tolist())
        # print(key)
        if key in self.eff_table:
            self.eff = self.eff_table[key]
        else:
            # self.eff = self.getEffofStructure(octave.double(struct_after.tolist()), self.wavelength, self.desired_angle)
            self.eff = self.getEffofStructure_benchmark(struct_after, self.wavelength, self.desired_angle)
            self.eff_table[key] = self.eff

        reward = (self.eff) ** 3
        # various kinds of reward can be set
        # reward = (result_after)**3.
        # reward = result_after - result_before
        # reward = 1-(1-result_after)**3

        self.struct = struct_after.copy()

        return struct_after.squeeze(), self.eff, reward, done

    def reset(self):  # initializing the env
        self.struct = np.ones(self.n_cells)
        eff_init = 0
        self.done = False
        if self.eff_table:
            with open(self.eff_file_path, 'wb') as f:
                json.dump(self.eff_table, f)
        return self.struct.squeeze(), eff_init

    def get_obs(self):
        return tuple(self.struct)

    def render(self, mode='human', close=False):
        plt.plot(self.struct)
