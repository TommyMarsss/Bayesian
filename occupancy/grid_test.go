package occupancy

import (
	"math"
	"testing"
)

func almostEq(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// LogOdds/Probability must round-trip and stay finite at the extremes.
func TestLogOddsRoundTrip(t *testing.T) {
	for _, p := range []float64{0.01, 0.1, 0.5, 0.9, 0.99} {
		if got := Probability(LogOdds(p)); !almostEq(got, p, 1e-12) {
			t.Errorf("round-trip p=%v got %v", p, got)
		}
	}
	for _, l := range []float64{-1000, -12, 0, 12, 1000} {
		p := Probability(l)
		if math.IsNaN(p) || math.IsInf(p, 0) || p < 0 || p > 1 {
			t.Errorf("Probability(%v) = %v, want finite in [0,1]", l, p)
		}
	}
}

// Requirement 1: saturation. Repeated identical observations must clamp at
// the configured bounds — never overflow, never NaN — and a single
// opposite observation must still move the belief by a full model step.
func TestSaturationNoOverflow(t *testing.T) {
	g := NewGrid(4, 4, DefaultLMin, DefaultLMax)
	m := DefaultSensorModel()
	hit := Ray{X0: 0, Y0: 0, X1: 3, Y1: 0} // endpoint (3,0)

	for i := 0; i < 100000; i++ {
		g.IntegrateScan([]Ray{hit}, m)
	}
	l, _ := g.LogOddsAt(3, 0)
	if l != DefaultLMax {
		t.Fatalf("after 1e5 hits l=%v, want clamped at %v", l, DefaultLMax)
	}
	p, _ := g.ProbAt(3, 0)
	if math.IsNaN(p) || math.IsInf(p, 0) || p >= 1 {
		t.Fatalf("saturated p=%v, want finite and < 1", p)
	}

	// One opposite observation must still have a well-defined effect:
	// exactly one LFree step down from the bound.
	g2 := NewGrid(4, 4, DefaultLMin, DefaultLMax)
	for i := 0; i < 100000; i++ {
		g2.IntegrateScan([]Ray{hit}, m)
	}
	g2.IntegrateScan([]Ray{{X0: 0, Y0: 0, X1: 2, Y1: 0}}, m) // (3,0) untouched
	g2.IntegrateScan([]Ray{{X0: 3, Y0: 0, X1: 0, Y1: 0}}, m) // (3,0) is origin -> free
	l2, _ := g2.LogOddsAt(3, 0)
	want := DefaultLMax + m.LFree()
	if !almostEq(l2, want, 1e-12) {
		t.Fatalf("after one opposite observation l=%v, want %v", l2, want)
	}

	// Symmetric: free saturation.
	g3 := NewGrid(4, 4, DefaultLMin, DefaultLMax)
	for i := 0; i < 100000; i++ {
		g3.IntegrateScan([]Ray{{X0: 3, Y0: 0, X1: 0, Y1: 0}}, m) // (3,0) free
	}
	l3, _ := g3.LogOddsAt(3, 0)
	if l3 != DefaultLMin {
		t.Fatalf("after 1e5 frees l=%v, want clamped at %v", l3, DefaultLMin)
	}
}

// Requirement 2: cells on the beam path are free, the endpoint is
// occupied, and the endpoint must not be degraded by any free update.
func TestRayFreeSpaceAndEndpoint(t *testing.T) {
	g := NewGrid(10, 10, DefaultLMin, DefaultLMax)
	m := DefaultSensorModel()
	g.IntegrateScan([]Ray{{X0: 1, Y0: 5, X1: 8, Y1: 5}}, m)

	for x := 1; x <= 7; x++ {
		if p, _ := g.ProbAt(x, 5); p >= 0.5 {
			t.Errorf("path cell (%d,5) p=%v, want < 0.5 (free)", x, p)
		}
	}
	pEnd, _ := g.ProbAt(8, 5)
	if pEnd <= 0.5 {
		t.Fatalf("endpoint p=%v, want > 0.5 (occupied)", pEnd)
	}
	// The endpoint received exactly the occupied update — no free update
	// from its own ray leaked in.
	if want := Probability(m.LOcc()); !almostEq(pEnd, want, 1e-12) {
		t.Fatalf("endpoint p=%v, want exactly %v (pure occupied update)", pEnd, want)
	}
}

// A cell that is one ray's endpoint but lies on another ray's free path
// must come out occupied: hit evidence wins within a scan.
func TestEndpointBeatsCrossingFreeRay(t *testing.T) {
	g := NewGrid(10, 10, DefaultLMin, DefaultLMax)
	m := DefaultSensorModel()
	rays := []Ray{
		{X0: 0, Y0: 0, X1: 5, Y1: 5}, // endpoint (5,5)
		{X0: 5, Y0: 9, X1: 5, Y1: 0}, // vertical ray passes through (5,5)
	}
	g.IntegrateScan(rays, m)
	p, _ := g.ProbAt(5, 5)
	if want := Probability(m.LOcc()); !almostEq(p, want, 1e-12) {
		t.Fatalf("contested cell p=%v, want %v (occupied wins)", p, want)
	}
}

// Requirement 3: an overconfident model must produce detectable anomalous
// convergence, while the sane model on identical data must not.
func TestOverconfidenceDetection(t *testing.T) {
	mkRays := func() []Ray {
		var rays []Ray
		for i := 0; i < 8; i++ {
			a := float64(i) * math.Pi / 4
			rays = append(rays, Ray{
				X0: 10, Y0: 10,
				X1: 10 + 6*math.Cos(a), Y1: 10 + 6*math.Sin(a),
			})
		}
		return rays
	}

	sane := NewGrid(21, 21, DefaultLMin, DefaultLMax)
	over := NewGrid(21, 21, DefaultLMin, DefaultLMax)
	detSane, detOver := NewDetector(), NewDetector()

	for s := 0; s < 3; s++ {
		sane.IntegrateScan(mkRays(), DefaultSensorModel())
		over.IntegrateScan(mkRays(), OverconfidentSensorModel())
		detSane.CheckScan(sane, s)
		detOver.CheckScan(over, s)
	}

	if len(detOver.Anomalies) == 0 {
		t.Fatal("overconfident model produced no anomalies after 3 scans")
	}
	for _, a := range detOver.Anomalies {
		if a.Obs > detOver.MaxObs {
			t.Errorf("anomaly %+v reported with too many observations", a)
		}
		if a.P > detOver.Lo && a.P < detOver.Hi {
			t.Errorf("anomaly %+v has non-extreme probability", a)
		}
	}
	if len(detSane.Anomalies) != 0 {
		t.Fatalf("sane model falsely flagged: %+v", detSane.Anomalies)
	}

	// The overconfident map should be saturated almost everywhere it
	// looked; the sane map mostly should not.
	_, satOver, fracOver := detOver.SaturatedFraction(over)
	_, satSane, fracSane := detSane.SaturatedFraction(sane)
	if satOver == 0 || fracOver < 0.5 {
		t.Errorf("overconfident map: %d saturated (frac %.2f), want majority", satOver, fracOver)
	}
	if fracSane > 0.2 {
		t.Errorf("sane map: %d saturated (frac %.2f), want small minority", satSane, fracSane)
	}
}

// Requirement 4 (convergence): long repeated observation drives beliefs to
// a stable fixed point at the saturation bounds, and conflicting
// observations keep the accumulator bounded forever.
func TestLongRunConvergence(t *testing.T) {
	g := NewGrid(10, 10, DefaultLMin, DefaultLMax)
	m := DefaultSensorModel()
	scan := []Ray{{X0: 1, Y0: 5, X1: 8, Y1: 5}}

	for i := 0; i < 200; i++ {
		g.IntegrateScan(scan, m)
	}
	pEnd, _ := g.ProbAt(8, 5)
	pFree, _ := g.ProbAt(4, 5)
	if want := Probability(DefaultLMax); !almostEq(pEnd, want, 1e-15) {
		t.Errorf("endpoint converged to %v, want %v", pEnd, want)
	}
	if want := Probability(DefaultLMin); !almostEq(pFree, want, 1e-15) {
		t.Errorf("free cell converged to %v, want %v", pFree, want)
	}

	// Fixed point: more scans change nothing.
	for i := 0; i < 50; i++ {
		g.IntegrateScan(scan, m)
	}
	pEnd2, _ := g.ProbAt(8, 5)
	pFree2, _ := g.ProbAt(4, 5)
	if pEnd != pEnd2 || pFree != pFree2 {
		t.Errorf("beliefs not stable: (%v,%v) -> (%v,%v)", pEnd, pFree, pEnd2, pFree2)
	}

	// Alternating hit/miss on the same cell stays bounded forever.
	g2 := NewGrid(4, 4, DefaultLMin, DefaultLMax)
	occ := []Ray{{X0: 0, Y0: 0, X1: 3, Y1: 0}}  // (3,0) occupied
	free := []Ray{{X0: 3, Y0: 0, X1: 0, Y1: 0}} // (3,0) free
	for i := 0; i < 10000; i++ {
		g2.IntegrateScan(occ, m)
		g2.IntegrateScan(free, m)
	}
	l, _ := g2.LogOddsAt(3, 0)
	if math.IsNaN(l) || l < DefaultLMin || l > DefaultLMax {
		t.Fatalf("alternating observations: l=%v outside [%v,%v]", l, DefaultLMin, DefaultLMax)
	}
}

// The simulator must run deterministically and the overconfident map must
// trigger anomalies on realistic noisy data.
func TestSimulate(t *testing.T) {
	res := Simulate(42, 72)
	if len(res.Frames) != 72 {
		t.Fatalf("got %d frames, want 72", len(res.Frames))
	}
	n := res.Width * res.Height
	for i, f := range res.Frames {
		if len(f.Sane) != n || len(f.Over) != n {
			t.Fatalf("frame %d has wrong size", i)
		}
	}
	if len(res.AnomOver) == 0 {
		t.Fatal("simulation: overconfident model produced no anomalies")
	}
	if len(res.AnomSane) != 0 {
		t.Fatalf("simulation: sane model falsely flagged %d anomalies", len(res.AnomSane))
	}
	// Determinism.
	res2 := Simulate(42, 72)
	if len(res2.AnomOver) != len(res.AnomOver) {
		t.Fatal("simulation not deterministic")
	}
}
