/* Explain actual 01 results in small reading steps; no policy learning occurs here. */
(() => {
  const envFile = 'src/environment/placement_env.py';
  const sessionFile = 'integration/visualization/learning_session.py';
  const env = (method, ...focus) => ({file: envFile, method, focus});
  const session = (method, ...focus) => ({file: sessionFile, method, focus});
  const frame = (title, paragraphs, code, extra = {}) => ({title, paragraphs, code, ...extra});
  const truth = value => value ? "True" : "False";

  function checkFrames(check, context) {
    const {observation: obs, x, y} = context;
    const name = obs.current_macro;
    const [width, height] = obs.circuit.blocks[name];
    if (check.kind === 'bounds') {
      const tests = [
        ["Why must the lower-left corner stay to the right of the canvas boundary", 'x >= 0', `${x} >= 0`, x >= 0,
          `>= means greater than or equal to. x locates the left edge of ${name}, while 0 is the canvas's left boundary. Requiring x >= 0 prevents the edge from crossing that boundary. Here, ${x >= 0 ? "the left edge stays inside" : "the left edge is outside"}, so the condition is ${truth(x >= 0)}.`, 'left'],
        ["Why must the lower-left corner stay above the bottom boundary", 'y >= 0', `${y} >= 0`, y >= 0,
          `y locates the bottom edge of ${name}, and 0 is the canvas bottom. Requiring y >= 0 prevents the edge from falling below the canvas. Here, ${y >= 0 ? "the bottom edge stays inside" : "the bottom edge is outside"}, so the condition is ${truth(y >= 0)}.`, 'bottom'],
        ["Why is the right edge x plus macro width", 'x + width <= self.width', `${x} + ${width} = ${check.right}; ${check.right} <= ${check.board_width}`, check.right <= check.board_width,
          `${name} starts at left edge x =  ${x} and extends right by ${width} length units, placing its right edge at ${check.right}.self.width = ${check.board_width} is the canvas right boundary. Macro width and canvas self.width are different quantities. <= means less than or equal to: the right edge may touch, but must not cross, this boundary.`, 'right'],
        ["Why is the top edge y plus macro height", 'y + height <= self.height', `${y} + ${height} = ${check.top}; ${check.top} <= ${check.board_height}`, check.top <= check.board_height,
          `${name} starts at bottom edge y =  ${y} and extends upward by ${height} length units, placing its top edge at ${check.top}.self.height = ${check.board_height} is the canvas top. Comparing the full rectangle's edges checks more than whether its lower-left corner is inside.`, 'top'],
      ];
      return [
        frame("First distinguish macro dimensions from canvas dimensions", [
          `The current macro ${name} has original width ${width} and height ${height}; the canvas width is ${check.board_width} and height is ${check.board_height}. Candidate (x, y) = ( ${x}, ${y}) identifies the macro's lower-left corner; its rectangle extends right and upward from that point.`,
          "Local width and height come from the current macro. self.width and self.height belong to the environment and describe the canvas. Next we check all four edges."], env('inspect_action', 'width, height = self.circuit["blocks"][self.current_macro]')),
        ...tests.map(([title, token, calculation, passed, explanation, edge]) => frame(title,
          [`Substitute the actual values: ${calculation}; the comparison evaluates to ${truth(passed)}.`, explanation],
          env('inspect_action', token), {edge, conditionFailed: !passed})),
        frame("Why are all four boundary conditions joined by and", [
          `The four results are ${truth(x >= 0)}, ${truth(y >= 0)}, ${truth(check.right <= check.board_width)}, ${truth(check.top <= check.board_height)}. and requires every condition to be True before inside becomes True.`,
          `For this candidate, inside =  ${truth(check.passed)}.${check.passed ? "The full rectangle is inside. The environment now checks overlap with placed macros; passing the boundary test alone has not established final legality." : "At least one edge is outside. The environment returns an error immediately and skips overlap checks."}`], env('inspect_action', 'inside =', 'if not inside:', 'return finish("Macro would extend outside the board")')),
      ];
    }
    const other = check.block;
    const p = obs.placements[other];
    const [ow, oh] = obs.circuit.blocks[other];
    const meanings = [
      ["Is the candidate's left edge left of the other rectangle's right edge", 'x, other_x + other_width', `${name}: left edge; ${other}: right edge`, 'left'],
      ["Is the other rectangle's left edge left of the candidate's right edge", 'other_x, x + width', `${other}: left edge; ${name}: right edge`, 'right'],
      ["Is the candidate's bottom edge below the other rectangle's top edge", 'y, other_y + other_height', `${name}: bottom edge; ${other}: top edge`, 'bottom'],
      ["Is the other rectangle's bottom edge below the candidate's top edge", 'other_y, y + height', `${other}: bottom edge; ${name}: top edge`, 'top'],
    ];
    return [
      frame(`Why compare the candidate with ${other} now`, [
        `Passing boundaries establishes that ${name} stays inside, but it must also avoid occupied space. ${other} is already placed at ( ${p.x}, ${p.y}), with dimensions ${ow} × ${oh}.`,
        `The candidate's x interval is [${x}, ${x + width}], and its y interval is [${y}, ${y + height}]; ${other} has x interval [${p.x}, ${p.x + ow}] and y interval [${p.y}, ${p.y + oh}]. The environment first retrieves these edge coordinates.`], env('inspect_action', 'for name, position in self.placements.items():', 'other_width, other_height =', 'other_x, other_y =')),
      ...meanings.map(([title, token, objects, edge], index) => {
        const [left, right] = check.comparisons[index];
        return frame(title, [
          `This compares ${objects}: ${left} < ${right}; the result is ${truth(left < right)}.`,
          `${index < 2 ? "The first two conditions together test positive-length intersection along x." : "The last two conditions together test positive-length intersection along y."}The source uses strict <. Equal edges make that comparison False and allow touching. One True comparison alone does not establish rectangle overlap.`], env('inspect_action', token), {edge});
      }),
      frame("Why does overlap require all four comparisons to be True", [
        `The four values are ${check.comparisons.map(([left, right]) => truth(left < right)).join(', ')}. all(...) tests whether every value is True; overlap =  ${truth(!check.passed)}. Positive-area overlap requires positive-length intersections in both x and y.`,
        `not reverses a boolean: True becomes False and False becomes True. passed = not overlap, so this overlap check ${check.passed ? "passed" : "failed"}.${check.passed ? "The environment continues checking other placed macros. It accepts the candidate after all comparisons are conflict-free." : `The environment reports an overlap with ${other}, so this candidate cannot execute.`}all can stop at the first False. The page unfolds the comparisons so that each decision remains inspectable.`], env('inspect_action', 'overlap = all(', '"passed": not overlap', 'if overlap:', 'return finish(f"Macro would overlap {name}")')),
    ];
  }

  function build(state) {
    const obs = state.observation, phase = state.phase, attempt = state.attempt;
    const transition = state.transitions.at(-1);
    const before = phase === 'apply' ? transition.observation : obs;
    const name = phase === 'apply' ? before.current_macro : obs.current_macro;
    const board = obs.board;
    const i = state.current.index ?? attempt?.index ?? 0;
    const x = attempt?.x ?? 0, y = attempt?.y ?? 0;
    const total = Object.keys(obs.circuit.blocks).length;
    const placed = Object.keys(obs.placements).length;
    if (phase === 'observe') return [
      frame("Does chapter 01 use RL or a fixed rule", [
        "Chapter 01 uses a handwritten policy: visit macros in input order, scan anchors row by row, and select the first legal anchor. It is a policy because it specifies action selection; it has no network and does not learn from reward.",
        "This step explains the environment, action and feedback using a replayable layout. Its completed layout is not automatically used as the PPO initial state; PPO episodes start on an empty canvas."], session('__init__', 'self.env = PlacementEnv(circuit, width, height)')),
      frame("What is on the canvas at the start of an episode", [
        "env is a PlacementEnv instance containing the circuit, canvas and placement state. reset() begins an episode: one placement attempt ending in completion or failure.",
        `reset() sets placements to an empty dictionary, initially containing no coordinates. This episode currently contains ${placed} placed macros: ${Object.keys(obs.placements).join(', ') || "none"}. Any existing positions come from earlier executed actions in this episode, rather than an automatically loaded initial floorplan.`], env('reset', 'self.placements = {}', 'self.reward = 0.0', 'self.terminated = False')),
      frame("Which macro is current and how is it selected", [
        `names preserves input-file order: ${Object.keys(obs.circuit.blocks).join(', ')}.len(placements) = ${placed}, meaning that ${placed} macros have been placed. Using that count as an index selects the next macro, ${name}.`,
        `Python indexes start at 0: names[0] is the first macro. We now need to place ${name}, whose original width and height are ${obs.circuit.blocks[name].join(' × ')}. A macro's position in this list differs from candidate-anchor index i.`], env('current_macro', 'return self.names[len(self.placements)]')),
      frame("What do state and action each describe", [
        "An observation describes the current situation: macro dimensions, nets, canvas, existing coordinates, current macro and remaining macros. An action specifies the next operation: an integer lower-left coordinate for the current macro.",
        `The current macro is ${name}, but no coordinate is selected yet. The next event proposes a candidate. Checking it does not write placements; a later step call executes the action.`], env('observation', '"circuit":', '"board":', '"placements":', '"current_macro":', '"remaining":')),
    ];
    if (phase === 'candidate') return [
      frame(`i = ${i} identifies which candidate`, [
        `i numbers candidate anchors for the current macro; the source calls it index. It is neither a macro ID nor a net ID. Each new macro restarts candidate numbering at 0.`,
        `At index = len(attempts), there are already ${i} candidate records, so the next index is ${i}.${i === 0 ? "0 is the first candidate under zero-based indexing. No earlier position has been checked." : `Earlier candidate indices are 0 through ${i - 1}; the next candidate is number ${i + 1} under one-based counting.`}`], session('_candidate', 'index = len(self.module["attempts"])')),
      frame("Why convert one index into two coordinates", [
        `Each canvas row contains ${board.width} integer anchors, with x ranging from 0 through ${board.width - 1}; y increases upward from 0 at the bottom. There are ${board.width * board.height} anchors in total. Macro size is not considered yet; the subsequent boundary checks remove anchors that cannot fit the full rectangle.`,
        `The convention moves right and then up a row: i = 0 maps to (0, 0), i =  ${board.width - 1} maps to ( ${board.width - 1}, 0), i = ${board.width} maps to (0, 1). This is a defined numbering convention, rather than coordinates predicted by a model.`], session('_candidate', 'x, y = index % self.env.width, index // self.env.width'), {mapping: true}),
      frame(`x = ${i} % ${board.width} evaluates to ${x}`, [
        `% calculates the remainder after dividing ${i} by ${board.width}. ${i} = ${Math.floor(i / board.width)} × ${board.width} + ${x}, so the remainder is ${x}. Every full row advances y; the remainder identifies the horizontal position in the current row.`,
        `Assigning the remainder to x gives x =  ${x}. The candidate lower-left corner is ${x} length units from the canvas left boundary. ${x === 0 ? "It is the leftmost anchor in this row, rather than a macro numbered zero." : `It is zero-based column ${x}, or anchor number ${x + 1} from the left under one-based counting.`}`], session('_candidate', 'index % self.env.width'), {edge: 'left', mapping: true}),
      frame(`y = ${i} // ${board.width} evaluates to ${y}`, [
        `// is integer division. These indices are nonnegative, so it returns the integer quotient. ${i} contains ${y} complete groups of ${board.width}, corresponding to ${y} complete preceding rows. That row number becomes y.`,
        `y = ${y} places the candidate lower-left corner ${y} length units above the canvas bottom. ${y === 0 ? "It is the bottom row because numbering begins at y = 0 at the bottom." : `It is row number ${y + 1} counted upward from the bottom.`}Both coordinates come from the same i: remainder gives x and integer division gives y.`], session('_candidate', 'index // self.env.width'), {edge: 'bottom', mapping: true}),
      frame(`Why check ( ${x}, ${y})`, [
        `Coordinates are written in (x, y) order. Combining x =  ${x}, y = ${y} gives ( ${x}, ${y}). This proposes placing the lower-left corner of ${name} at that position.`,
        "inspect_action(x, y) is an environment method. It checks boundaries and overlap and returns a report without placing the macro. The following explanations unfold that same actual report."], session('_candidate', 'report = self.env.inspect_action(x, y)'), {mapping: true}),
    ];
    if (phase === 'bounds' || phase === 'overlap') return checkFrames(state.current.check, {observation: obs, x, y});
    if (phase === 'mask') {
      const legal = attempt.report.legal;
      const overlaps = attempt.report.checks.filter(check => check.kind === 'overlap');
      return [
        frame("What is mask and what does its index identify", [
          `mask is a boolean table ordered by candidate index. mask[${i}] accesses candidate ${i}, at ( ${x}, ${y}). Indexing starts at 0, so mask[0] is the first entry. True permits execution; False forbids it.`,
          "JSON displays Python True and False as true and false with the same meaning. Unfolded-but-unknown entries are None and displayed as Unknown. Unknown is different from False and is not used to select an action.",
          "The environment's legal_mask() can generate the whole boolean grid. The teaching session separately records which checks have been explained, filling entries as reading advances. Both use the same boundary and overlap rules."], session('_begin_module', '"mask": [None] *'), {mask: true}),
        frame("How is this candidate's final legality established", [
          `The boundary check ${attempt.report.checks[0].passed ? "passed" : "failed"}.${overlaps.length ? `It then compared ${overlaps.map(check => check.block).join(', ')}, ${legal ? "none of these macros overlaps the candidate" : "the environment returned an error upon finding a conflict"}.` : (legal ? "There are no placed macros. The for loop has no objects to compare, so no overlap error occurs." : "The boundary test failed, so overlap checks were skipped.")}`,
          legal ? "Execution reaches return finish() without an error argument. finish defaults error to None. Therefore error is None evaluates to True, and report[\"legal\"] is True."
            : `The check returned early through finish(error) with error:  ${attempt.report.error}. error is not None, so error is None evaluates to False and report["legal"] is False.`], env('inspect_action', 'def finish(error=None):', '"legal": error is None', 'return finish()'), {mask: true}),
        frame(`mask[${i}] = ${legal ? 'True' : 'False'} is written by which statement`, [
          `report is the dictionary returned by the check. report["legal"] retrieves its legal field. The session writes the actual boolean ${truth(legal)} into mask entry ${i} for the current macro. That value comes from the boundary and overlap checks, rather than randomness or a guess based on Area.`,
          `The entry is now mask[${i}] = ${legal ? 'True' : 'False'}. resolved = True means this entry's explanation is complete. It differs from legal: a fully checked candidate can still be illegal.`], session('advance', 'self.module["mask"][attempt["index"]] = attempt["report"]["legal"]', 'attempt["resolved"] = True'), {mask: true}),
      ];
    }
    if (phase === 'reject') return [
      frame("Why does rejecting a candidate leave placement unchanged", [
        `mask[${i}] is False because of ${attempt.report.error.includes('overlap') ? `overlap with ${attempt.report.error.split(' ').at(-1)} ` : "the full rectangle extends outside the canvas"}. The session enters reject without calling env.step, so placements stays unchanged.`,
        "The candidate record is retained for inspection. Proposing and checking a position are separate from executing an action."], session('advance', 'self.phase = "choose" if self.module["mask"][attempt["index"]] else "reject"')),
      frame(`Why is the next index ${i + 1}`, [
        `The current candidate has been appended to attempts, changing its length from ${i} to ${i + 1}. On the next _candidate() call, len(attempts) supplies index ${i + 1}. The source does not need a separate i += 1 statement.`,
        `Apply the same conversion: x =  ${i + 1} % ${board.width} = ${(i + 1) % board.width}, y = ${i + 1} // ${board.width} = ${Math.floor((i + 1) / board.width)}. The next position checked is ( ${(i + 1) % board.width}, ${Math.floor((i + 1) / board.width)}), rather than a jump to a position already known to fit.`], session('_candidate', 'index = len(self.module["attempts"])', 'self.module["attempts"].append(attempt)', 'x, y = index % self.env.width, index // self.env.width'), {mapping: true}),
    ];
    if (phase === 'choose') return [
      frame("Why does a permitted position become the selected position", [
        `Legality answers whether a placement may execute. A policy chooses among legal options. Chapter 01 defines its rule in advance: visit increasing indices and select the first True. It does not optimize Area or HPWL across all legal positions.`,
        "This is a fixed decision rule. Receiving a reward does not change it; no learned network selects positions in this chapter."], session('advance', 'self.phase = "choose" if self.module["mask"][attempt["index"]] else "reject"')),
      frame("Why is this the first legal position", [
        i ? `Indices 0 through ${i - 1} were checked in order and marked False. The current mask[${i}] is True. No earlier candidate passed, making this the first legal anchor in scan order.`
          : "i = 0 is the first candidate of this search. It passed and was marked True, so it is immediately the first legal anchor.",
        `First refers to index order, rather than minimum Area or HPWL. The fixed rule selects ( ${x}, ${y}) and stops this search.`], session('advance', 'if self.phase == "choose":', 'self.module["chosen"] = [attempt["x"], attempt["y"]]'), {mask: true}),
      frame("Why has the canvas not changed after choosing an action", [
        `chosen now stores [${x}, ${y}], the action intended for the environment. This assignment changes only the session's selection record; it does not write environment placements.`,
        "The next choose branch calls env.step(action). Pausing here separates the reason for selecting an action from executing it."], session('advance', 'self.module["chosen"] =', 'action = self.module["chosen"]', 'observation, reward, terminated, truncated, info = self.env.step(action)')),
    ];
    if (phase === 'apply') {
      const [ax, ay] = transition.action;
      return [
        frame("What does env mean in env.step(action)", [
          "env is the PlacementEnv instance holding this episode's circuit and placement state. step is a method defined with def in that class. env.step(...) asks this environment to execute one action.",
          "These reading frames explain one actual call. The canvas first shows its before-state, then shows the after-state when the coordinate write is explained. Reading the text does not execute another action."], env('step', 'def step(self, action):'), {view: 'before'}),
        frame("What does action contain and what does self refer to", [
          `The action is [${ax}, ${ay}], containing two integer coordinates. The environment already identifies macro ${name}, so the action does not need to include its name.`,
          `In env.step(action), Python passes env as self automatically. x, y = action unpacks its two values into x =  ${ax}, y = ${ay}. This is unpacking, rather than another coordinate search.`], env('step', 'def step(self, action):', 'x, y = action'), {view: 'before'}),
        frame("Why does step validate an already selected position again", [
          `step calls action_error(${ax}, ${ay}) to confirm legality at execution time. action_error calls the same inspect_action, so execution and selection share boundary and overlap rules.`,
          "An error string triggers raise ValueError(error), stopping the call before any coordinate is written. This action has no error and continues. A caller invoking step directly must satisfy the same constraints."], env('step', 'error = self.action_error(x, y)', 'if error:', 'raise ValueError(error)'), {view: 'before'}),
        frame("Which statement actually places the macro", [
          `Before writing, self.current_macro is ${name}. The assignment adds ${name}: {x: ${ax}, y: ${ay}, rotated: False} to placements, changing the placed count from ${Object.keys(before.placements).length} to ${placed}.`,
          "rotated: False keeps the input orientation. The canvas now displays the real after-state. The assignment changed the coordinates; the browser did not invent a placement animation from a check result."], env('step', 'self.placements[self.current_macro] ='), {view: 'after'}),
        frame("What happens after coordinates are written", [
          "self.reward = 0.0 resets this action's feedback so it does not reuse the previous reward. _finish_if_needed() then checks completion or an impossible continuation and overwrites reward on termination.",
          `The actual call returned reward =  ${transition.reward}, terminated = ${transition.terminated}. Later reward explanations inspect each branch and number; this frame establishes the order inside step.`], env('step', 'self.reward = 0.0', 'self._finish_if_needed()'), {view: 'after'}),
        frame("What are the five results returned by step", [
          `observation is the after-state. reward is this action's feedback, ${transition.reward}; terminated = ${transition.terminated} indicates natural termination. truncated =  ${transition.truncated} indicates an external cutoff; this environment currently always returns False for truncation.`,
          "info contains metrics and the termination reason. The caller unpacks the five values in order and records the before/after states in a transition for later inspection."], env('step', 'return self.observation(), self.reward, self.terminated, False, self.info()'), {view: 'after'}),
      ];
    }
    if (phase === 'metrics') {
      const details = state.details, b = details.bounds, m = details.metrics;
      return [
        frame("Why is occupied Area different from the entire canvas Area", [
          "64 by 64 defines the allowed placement region. The objective measures the tight axis-aligned bounding box enclosing the occupied rectangles. More compact layouts generally reduce this occupied Area.",
          `The environment finds the leftmost, bottommost, rightmost and topmost occupied edges. Here, left =  ${b.left}, bottom = ${b.bottom}, right = ${b.right}, top = ${b.top}. Right and top edges include each macro's width and height.`], env('metric_details', '"left": min(', '"bottom": min(', '"right": max(', '"top": max(')),
        frame("How do bounding-box width and height produce Area", [
          `Width = rightmost edge - leftmost edge =  ${b.right} − ${b.left} = ${b.right - b.left}; height = topmost edge - bottommost edge =  ${b.top} − ${b.bottom} = ${b.top - b.bottom}.`,
          `Multiplying gives Area =  ${b.right - b.left} × ${b.top - b.bottom} = ${m.area}. Only placed macros contribute, so this is a partial-layout bounding box until completion. Translating the whole layout leaves both differences unchanged.`], env('metrics', 'area = (right - left) * (top - bottom)')),
        ...details.nets.map(net => {
          const points = Object.entries(net.placed), xs = points.map(([, p]) => p.x), ys = points.map(([, p]) => p.y);
          return frame(`Net ${net.index} contributes how much HPWL`, [
            `The currently placed endpoints of this net are ${points.map(([n, p]) => `${n}(${p.x}, ${p.y})`).join(', ') || "none"}. Endpoints use macro lower-left corners, rather than centers. ${net.missing.length ? `Unplaced endpoints ${net.missing.join(', ')} are excluded and are not assigned fictitious coordinates (0, 0).` : "Every endpoint is placed."}`,
            points.length ? `The x-coordinate list is [${xs.join(', ')}], giving span max(x) - min(x) =  ${Math.max(...xs)} − ${Math.min(...xs)} = ${net.dx}. The y-coordinate list is [${ys.join(', ')}], giving span ${Math.max(...ys)} − ${Math.min(...ys)} = ${net.dy}. Their sum is HPWL =  ${net.hpwl}.`
              : "With no placed endpoints, the current contribution is 0. With one endpoint, maximum and minimum coincide, so the span is also 0.",
            "HPWL adds the endpoint bounding-box width and height as a connection-length estimate. It does not generate an actual routed wire."], env('metric_details', 'placed = {name:', 'xs =', 'ys =', 'dx =', 'dy =', '"hpwl": dx + dy'));
        }),
        frame("What does summing the contribution of every net mean", [
          `The net contributions are ${details.nets.map(net => net.hpwl).join(', ') || "no nets"}, summing to ${m.hpwl}.${m.complete ? "All macros have coordinates, so this is the complete HPWL for this layout." : "Some macros are unplaced, so the quantity describes only the known endpoints. Later placements may increase it."}`,
          "These metrics are stored in info[\"metrics\"]. Metrics describe geometry; reward design specifies when and how that geometry becomes feedback."], env('metrics', 'hpwl +=', 'return {"area": area, "hpwl": hpwl')),
      ];
    }
    if (phase === 'reward') {
      const m = state.info.metrics, event = state.current;
      const finished = state.info.reason === 'complete';
      const frames = [
        frame("Is reward = 0 a geometric result or a design choice", [
          "Reward is the numerical feedback returned for this action. It does not automatically equal Area or HPWL. The developer defines how outcomes become feedback.",
          "This environment uses terminal feedback: intermediate legal steps return 0; complete layouts receive negative normalized Area/HPWL cost; impossible continuations receive -2. Other RL environments may use intermediate rewards."], env('step', 'self.reward = 0.0', 'self._finish_if_needed()')),
        frame("How does the environment recognize completion", [
          `There are currently ${placed} / ${total} placed macros. current_macro uses len(placements) to determine whether another macro remains. It returns None after all macros are placed.`,
          `Here current_macro =  ${obs.current_macro ?? 'None'}, so if self.current_macro is None evaluates to ${truth(obs.current_macro === null)}.${finished ? "The completion branch sets terminated to True and computes terminal feedback." : "The completion branch is skipped, and the environment next checks whether continuation is possible."}`], env('_finish_if_needed', 'if self.current_macro is None:', 'self.terminated, self.reason = True, "complete"')),
      ];
      if (!finished) frames.push(frame("Why check continuation before all macros are placed", [
        "Earlier fixed choices can block the next macro. Remaining macros do not ensure another action exists. legal_mask() checks every anchor in the updated state. any(mask) is True if at least one action is possible; not any(mask) means none is possible.",
        `Here terminated =  ${state.info.terminated}, reason = ${state.info.reason ?? 'None'}.${state.info.reason === 'no_legal_action' ? "No legal continuation exists. The environment enters the failure branch, sets reward to -2 and terminates the episode." : "The failure branch was not entered, so a legal continuation exists. _finish_if_needed leaves the reset reward at 0. Area was still computed, and this zero reward does not identify an optimal action."}`], env('_finish_if_needed', 'elif not any(self.legal_mask()):', 'self.terminated, self.reason, self.reward = True, "no_legal_action", -2.0')));
      if (finished) frames.push(
        frame("Why divide terminal metrics by reference quantities", [
          `The actual Area is ${m.area} and HPWL is ${m.hpwl}. Their units and scales differ. The formula divides each by its reference: canvas Area ${board.width} × ${board.height} = ${event.area_reference} and max(1, net count * (canvas width + height)) =  ${event.hpwl_reference}. max(1, ...) avoids division by zero.`,
          "The environment explicitly defines these denominators and equal weights of 0.5. They are normalization references, rather than dataset-provided optimal metrics."], env('_finish_if_needed', 'area_reference =', 'hpwl_reference =')),
        frame("Why negate the cost and what is the actual reward", [
          `Area term = -0.5 *  ${m.area} / ${event.area_reference} = ${-0.5 * m.area / event.area_reference}; HPWL term = -0.5 *  ${m.hpwl} / ${event.hpwl_reference} = ${-0.5 * m.hpwl / event.hpwl_reference}. Adding them gives ${state.reward}.`,
          "Negating cost makes smaller Area or HPWL produce a reward closer to zero and therefore numerically larger. PPO can optimize larger expected return. The fixed scan rule in this chapter does not change its decisions using reward."], env('_finish_if_needed', 'self.reward = (-self.area_weight *')),
      );
      frames.push(frame("Why can reward remain zero after Area and HPWL are calculated", [
        `info retains current Area ${m.area} and HPWL ${m.hpwl}; the reward branch returns ${state.reward}. Recording geometry and choosing feedback occur in different logic, so nonzero metrics can accompany zero reward.`,
        "step returns both together. Understanding this base contract lets us later reason about terminal rewards, intermediate rewards or reward shaping while keeping their causes explicit."], env('step', 'return self.observation(), self.reward, self.terminated, False, self.info()')));
      return frames;
    }
    if (phase === 'next_observation') return [
      frame("What changed in s_next", [
        `The next state contains ${placed} placed macros and their actual lower-left coordinates. Circuit dimensions and nets are unchanged; the latest action added one placement.`,
        `current_macro is now ${obs.current_macro ?? 'None'} and remaining is ${obs.remaining.join(', ') || "an empty list"}. The environment constructs these fields from the new placement count; the model does not guess them again.`], env('observation', '"placements":', '"current_macro":', '"remaining":')),
      frame("Can a caller corrupt the environment through its returned observation", [
        "observation() returns deep copies of the circuit and coordinates. Editing the returned data does not directly mutate the environment's internal placements.",
        "Use step(action) to change the environment so that the same legality and feedback rules remain enforced."], env('observation', '"circuit": deepcopy(self.circuit)', '"placements": deepcopy(self.placements)')),
      frame("Why regenerate the mask for each macro", [
        obs.current_macro ? `The next macro is ${obs.current_macro}. Its dimensions may differ, and another obstacle is now present. The same coordinate may be illegal for the next macro even if it was legal earlier, so the old mask cannot simply be reused.` : "There is no next macro. The episode is over, so no new candidate mask is constructed and step cannot place another macro.",
        obs.current_macro ? "The next event creates a fresh reading record, marks mask entries Unknown and restarts at candidate index 0. For example, B0 may fit at (0,0), but B1 at (0,0) overlaps B0 after it is placed." : "The next event explains termination; earlier actions and feedback remain inspectable."], session('_begin_module', '"mask": [None] *', 'self._event(self.phase, observation=self.env.observation())')),
    ];
    return [
      frame("What did this finished episode accomplish", [
        state.info.reason === 'complete' ? `This fixed-policy trajectory placed all ${total} macros, with Area ${state.info.metrics.area}, complete HPWL ${state.info.metrics.hpwl} and terminal reward ${state.reward}.`
          : `The episode ended without a legal continuation after placing ${placed} / ${total} macros. A fixed policy can fail this way; reward =  ${state.reward}.`,
        "An episode is one full-circuit placement attempt; a step is one action inside it. This trajectory validates the environment and feedback without updating network parameters or automatically initializing PPO with the result."], env('step', 'return self.observation(), self.reward, self.terminated, False, self.info()')),
      frame("What is the complete causal sequence for one action", [
        "Environment state -> fixed-policy candidate index -> boundary and overlap checks -> boolean mask -> first legal selection -> step rechecks and writes coordinates -> metrics and termination -> reward and next state.",
        "Checking, selecting, executing and scoring have distinct source statements. Chapter 02 separately replaces fixed selection with actor/critic decisions after these environment rules are understood."], session('advance', 'action = self.module["chosen"]', 'self.env.step(action)')),
    ];
  }
  window.EE680PlacementLesson = {build, checkFrames};
})();
