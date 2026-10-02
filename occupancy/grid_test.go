package occupancy

import (
	"math"
	"testing"
)

var saneModel = NewSensorModel(0.70, 0.40)

// 1. Saturation: the accumulator must never overflow, and a saturated
// cell must still respond to contradictory evidence within a bounded
// number of observations.
func TestSaturationBoundsAndRecovery(t *testing.T) {
	g := NewGrid(4, 4, 1.0, DefaultLMin, DefaultLMax)

	// Hammer one cell with occupied updates far beyond what saturation allows.
	for i := 0; i < 100000; i++ {
		g.UpdateRay(0, 0, Ray{EndX: 2, EndY: 2, Hit: true}, saneModel)
	}
	l := g.LogOdds(2, 2)
	if math.IsNaN(l) || math.IsInf(l, 0) {
		t.Fatalf("log-odds not finite after 1e5 updates: %v", l)
	}
	if l != DefaultLMax {
		t.Fatalf("expected saturation at LMax=%v, got %v", DefaultLMax, l)
	}
	p := g.Probability(2, 2)
	if math.IsNaN(p) || p <= 0.5 || p >= 1 {
		t.Fatalf("probability at saturation should be in (0.5, 1), got %v", p)
	}

	// A single contradictory observation must move the accumulator by
	// exactly LFree — saturation must not swallow counter-evidence.
	before := g.LogOdds(2, 2)
	g.addDelta(2, 2, saneModel.LFree)
	if got := g.LogOdds(2, 2); math.Abs(got-(before+saneModel.LFree)) > 1e-12 {
		t.Fatalf("single free update at saturation moved cell by %v, want %v",
			got-before, saneModel.LFree)
	}

	// Recovery to "more likely free than occupied" must take a bounded
	// number of free observations: at most ceil(LMax/|LFree|)+1.
	maxSteps := int(math.Ceil(DefaultLMax/math.Abs(saneModel.LFree))) + 1
	steps := 0
	for g.Probability(2, 2) >= 0.5 && steps < maxSteps+10 {
		g.addDelta(2, 2, saneModel.LFree)
		steps++
	}
	if g.Probability(2, 2) >= 0.5 {
		t.Fatalf("cell never recovered from saturation after %d free updates", steps)
	}
	if steps > maxSteps {
		t.Fatalf("recovery took %d free updates, want <= %d (saturation bound ineffective?)", steps, maxSteps)
	}

	// Same for the free-space direction.
	g2 := NewGrid(4, 4, 1.0, DefaultLMin, DefaultLMax)
	for i := 0; i < 100000; i++ {
		g2.addDelta(1, 1, saneModel.LFree)
	}
	if l := g2.LogOdds(1, 1); l != DefaultLMin {
		t.Fatalf("expected saturation at LMin=%v, got %v", DefaultLMin, l)
	}
	if p := g2.Probability(1, 1); math.IsNaN(p) || p <= 0 || p >= 0.5 {
		t.Fatalf("probability at LMin should be in (0, 0.5), got %v", p)
	}
}

// Probability conversion must be numerically stable for extreme inputs.
func TestProbabilityStability(t *testing.T) {
	for _, l := range []float64{-1000, -100, 0, 100, 1000} {
		p := ProbabilityOf(l)
		if math.IsNaN(p) || p < 0 || p > 1 {
			t.Fatalf("ProbabilityOf(%v) = %v, out of [0,1]", l, p)
		}
	}
	if p := ProbabilityOf(1000); p != 1 {
		t.Fatalf("ProbabilityOf(1000) = %v, want 1", p)
	}
	if p := ProbabilityOf(-1000); p != 0 {
		t.Fatalf("ProbabilityOf(-1000) = %v, want 0", p)
	}
}

// 2. Ray tracing: traversed cells become free, the endpoint becomes
// occupied, and the endpoint must not be clobbered by free-space updates.
func TestRayFreeSpaceAndEndpoint(t *testing.T) {
	g := NewGrid(20, 20, 1.0, DefaultLMin, DefaultLMax)
	g.UpdateRay(2, 2, Ray{EndX: 12, EndY: 2, Hit: true}, saneModel)

	for x := 2; x <= 11; x++ {
		if l := g.LogOdds(x, 2); l >= 0 {
			t.Fatalf("traversed cell (%d,2) should be free (l<0), got %v", x, l)
		}
	}
	if l := g.LogOdds(12, 2); l <= 0 {
		t.Fatalf("endpoint (12,2) should be occupied (l>0), got %v", l)
	}
	if l := g.LogOdds(5, 5); l != 0 {
		t.Fatalf("untouched cell should stay unknown (l=0), got %v", l)
	}

	// A miss (no hit) marks even the final cell as free.
	g2 := NewGrid(20, 20, 1.0, DefaultLMin, DefaultLMax)
	g2.UpdateRay(2, 2, Ray{EndX: 12, EndY: 2, Hit: false}, saneModel)
	if l := g2.LogOdds(12, 2); l >= 0 {
		t.Fatalf("miss ray endpoint should be free (l<0), got %v", l)
	}
}

// Within one scan, a cell that is one beam's endpoint must stay occupied
// even when another beam of the same scan passes through it.
func TestScanEndpointNotOverwrittenByOtherRay(t *testing.T) {
	g := NewGrid(30, 30, 1.0, DefaultLMin, DefaultLMax)
	origin := [2]int{15, 15}
	end := [2]int{20, 15} // beam A hits here
	rays := []Ray{
		{EndX: end[0], EndY: end[1], Hit: true},
		{EndX: 28, EndY: 15, Hit: false}, // beam B passes straight through `end`
	}
	g.UpdateScan(origin[0], origin[1], rays, saneModel)
	if l := g.LogOdds(end[0], end[1]); l <= 0 {
		t.Fatalf("endpoint cell %v was overwritten by free-space update, l=%v", end, l)
	}
	// Sanity: a purely traversed neighbor is free.
	if l := g.LogOdds(18, 15); l >= 0 {
		t.Fatalf("traversed cell (18,15) should be free, got %v", l)
	}
}

// 3. Overconfidence: an overconfident inverse model pushes cells to
// extreme probabilities after very few noisy observations; Analyze must
// flag them, while a sane model must not be flagged.
func TestOverconfidenceDetection(t *testing.T) {
	overconfident := NewSensorModel(0.99, 0.01)

	// A handful of noisy, partly contradictory observations of one cell.
	// Alternating hit/miss evidence ending on a hit: with an overconfident
	// model the last single observation alone decides the cell's belief.
	noisyUpdates := func(g *Grid, x, y int, m SensorModel) {
		deltas := []float64{m.LOcc, m.LFree, m.LOcc, m.LFree, m.LOcc}
		for _, d := range deltas {
			g.addDelta(x, y, d)
		}
	}

	gBad := NewGrid(8, 8, 1.0, DefaultLMin, DefaultLMax)
	noisyUpdates(gBad, 4, 4, overconfident)
	stats := gBad.Analyze(DefaultMinObs, DefaultExtreme)
	if stats.Premature == 0 {
		t.Fatalf("overconfident model: expected premature extreme cells, got none "+
			"(p=%.4f after 5 noisy observations)", gBad.Probability(4, 4))
	}
	if cells := gBad.PrematureCells(DefaultMinObs, DefaultExtreme); len(cells) == 0 {
		t.Fatal("PrematureCells returned nothing for overconfident model")
	}

	gSane := NewGrid(8, 8, 1.0, DefaultLMin, DefaultLMax)
	noisyUpdates(gSane, 4, 4, saneModel)
	if stats := gSane.Analyze(DefaultMinObs, DefaultExtreme); stats.Premature != 0 {
		t.Fatalf("sane model falsely flagged: %+v (p=%.4f)", stats, gSane.Probability(4, 4))
	}
	if p := gSane.Probability(4, 4); p > DefaultExtreme || p < 1-DefaultExtreme {
		t.Fatalf("sane model reached extreme p=%.4f after 5 noisy observations", p)
	}
}

// Overconfidence at the map level: simulate noisy scans and confirm the
// detector fires for the overconfident model but not for the sane one.
func TestOverconfidenceDetectionInSimulation(t *testing.T) {
	world := RoomWorld(48, 36)
	poses := []Pose{{X: 6, Y: 4.5}, {X: 6.5, Y: 4.5}, {X: 7, Y: 4.5}}
	base := SimConfig{
		Resolution: 0.25, LMin: DefaultLMin, LMax: DefaultLMax,
		MaxRange: 5, Beams: 36, NoiseSigma: 0.15, SpuriousProb: 0.05, Seed: 7,
	}

	cfg := base
	cfg.Model = NewSensorModel(0.99, 0.01)
	framesBad, _ := RunScenario(world, poses, cfg)
	if got := framesBad[len(framesBad)-1].Stats.Premature; got == 0 {
		t.Fatal("simulation with overconfident model produced no premature extremes")
	}

	cfg.Model = saneModel
	framesOK, _ := RunScenario(world, poses, cfg)
	if got := framesOK[len(framesOK)-1].Stats.Premature; got != 0 {
		t.Fatalf("simulation with sane model flagged %d premature extremes", got)
	}
}

// 4. Convergence: under consistent observations the map converges to a
// stable probability and stays there — no drift, no oscillation.
func TestConvergenceToStableValue(t *testing.T) {
	g := NewGrid(4, 4, 1.0, DefaultLMin, DefaultLMax)

	prev := g.Probability(2, 2)
	for i := 0; i < 500; i++ {
		g.addDelta(2, 2, saneModel.LOcc)
		p := g.Probability(2, 2)
		if p < prev {
			t.Fatalf("probability decreased under consistent occupied evidence: %v -> %v", prev, p)
		}
		prev = p
	}
	want := ProbabilityOf(DefaultLMax)
	if math.Abs(prev-want) > 1e-12 {
		t.Fatalf("converged to %v, want saturation limit %v", prev, want)
	}
	// Further observations change nothing — the value is stable.
	for i := 0; i < 100; i++ {
		g.addDelta(2, 2, saneModel.LOcc)
	}
	if p := g.Probability(2, 2); p != prev {
		t.Fatalf("probability drifted after convergence: %v -> %v", prev, p)
	}

	// Mixed evidence at a fixed ratio converges to a bounded equilibrium:
	// the accumulator can never leave [LMin, LMax], so the probability
	// stays within the corresponding bounds forever.
	g2 := NewGrid(4, 4, 1.0, DefaultLMin, DefaultLMax)
	for i := 0; i < 10000; i++ {
		if i%3 == 0 {
			g2.addDelta(1, 1, saneModel.LOcc)
		} else {
			g2.addDelta(1, 1, saneModel.LFree)
		}
		l := g2.LogOdds(1, 1)
		if l < DefaultLMin || l > DefaultLMax {
			t.Fatalf("accumulator escaped bounds under mixed evidence: %v", l)
		}
	}
}
