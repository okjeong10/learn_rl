import torchcraft_py.torchcraft as tc
import torchcraft_py.proto as proto


class FindGoal(object):
    def __init__(self):
        ip = 'localhost'
        port = '11111'

        self.action = ['left', 'right', 'up', 'down']
        self.client = tc.Client(ip, port)
        self.ore = 0
        self.client.connect()

        setup = [proto.concat_cmd(proto.commands['set_speed'], 60),
                 proto.concat_cmd(proto.commands['set_gui'], 1),
                 proto.concat_cmd(proto.commands['set_frameskip'], 9),
                 proto.concat_cmd(proto.commands['set_cmd_optim'], 1)]

        self.client.send(setup)

        while True:
            self.client.receive()
            units = self.client.state.d['units_myself']
            if units:
                break

    def _wait_for_restart(self):
        while True:
            self.client.receive()
            units = self.client.state.d['units_myself']
            if units:
                break

    def _make_command(self, action):
        units = self.client.state.d['units_myself']
        if not units:
            return ""

        uid, ut = units.items()[0]
        mov_pos = [ut.x, ut.y]
        if action == 'left':
            mov_pos[0] -= 8
        elif action == 'right':
            mov_pos[0] += 8
        elif action == 'up':
            mov_pos[1] -= 8
        elif action == 'down':
            mov_pos[1] += 8

        cmd = [proto.concat_cmd(proto.commands['command_unit'],
                               uid, proto.unit_command_types['Move'], -1,
                               mov_pos[0], mov_pos[1], -1),
               ]

        return cmd, mov_pos

    def get_reward(self, action):
        command, mov_pos = self._make_command(action)
        self.client.send(command)
        be_killed = False
        unit_x = -1
        unit_y = -1
        while True:
            self.client.receive()
            units = self.client.state.d['units_myself']
            if units:
                unit_x = units.values()[0].x
                unit_y = units.values()[0].y
                if unit_x == mov_pos[0] and unit_y == mov_pos[1]:
                    break
            else:
                be_killed = True
                break

        if be_killed:
            self._wait_for_restart()
            self.client.receive()
            new_ore = self.client.state.d['frame'].resources[0].ore
            reward = new_ore - self.ore
            self.ore = new_ore
        else:
            reward = 0

        return [unit_x, unit_y], reward, be_killed

    def state(self):
        units = self.client.state.d['units_myself']
        unit_x = -1
        unit_y = -1
        if units:
            unit_x = units.values()[0].x
            unit_y = units.values()[0].y

        return [unit_x, unit_y]


def test():
    find_goal = FindGoal()
    actions = ['right', 'right', 'down', 'right', 'right', 'right', 'right',
               'right', 'right', 'right']
    r_sum = 0
    while True:
        for a in actions:
            s1, r, end = find_goal.get_reward(a)
            r_sum += r
            if end is True:
                break


if __name__=='__main__':
    test()
