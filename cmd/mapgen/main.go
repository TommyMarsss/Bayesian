// Command mapgen runs the occupancy-grid simulation and writes a single
// self-contained HTML file (map.html) that replays the map updates.
package main

import (
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/TommyMarsss/Bayesian/occupancy"
)

//go:embed template.html
var template string

type payload struct {
	W        int                 `json:"w"`
	H        int                 `json:"h"`
	World    string              `json:"world"`
	Frames   []string            `json:"frames"`
	AnomSane []occupancy.Anomaly `json:"anomSane"`
	AnomOver []occupancy.Anomaly `json:"anomOver"`
}

func main() {
	res := occupancy.Simulate(42, 72)

	p := payload{W: res.Width, H: res.Height, AnomSane: res.AnomSane, AnomOver: res.AnomOver}
	p.World = base64.StdEncoding.EncodeToString(res.World)
	for _, f := range res.Frames {
		buf := make([]byte, 0, len(f.Sane)+len(f.Over))
		buf = append(buf, f.Sane...)
		buf = append(buf, f.Over...)
		p.Frames = append(p.Frames, base64.StdEncoding.EncodeToString(buf))
	}

	js, err := json.Marshal(p)
	if err != nil {
		fmt.Fprintln(os.Stderr, "marshal:", err)
		os.Exit(1)
	}
	html := strings.Replace(template, "/*__DATA__*/", string(js), 1)
	if err := os.WriteFile("map.html", []byte(html), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote map.html (%d frames, %d overconfidence anomalies detected)\n",
		len(res.Frames), len(res.AnomOver))
}
