package main

import (
	"database/sql"
	"log"
	"net/http"
	"os"
	"sync/atomic"

	"example.com/m/handler"
	"example.com/m/internal/database"
	"github.com/joho/godotenv"
	_ "github.com/lib/pq"
)

func main() {
	const filepathRoot = "."
	const port = "8080"

	godotenv.Load()
	dbURL := os.Getenv("DB_URL")
	if dbURL == "" {
		log.Fatal("DB_URL must be set")
	}
	Platform := os.Getenv("PLATFORM")
	if Platform == "" {
		log.Fatal("PLATFORM must be set")
	}
	JwtSecret := os.Getenv("JWT_SECRET")
	if JwtSecret == "" {
		log.Fatal("JWT_SECRET environment variable is not set")
	}
	PolkaKey := os.Getenv("POLKA_KEY")
	if PolkaKey == "" {
		log.Fatal("POLKA_KEY environment variable is not set")
	}

	dbConn, err := sql.Open("postgres", dbURL)
	if err != nil {
		log.Fatalf("Error opening database: %s", err)
	}
	dbQueries := database.New(dbConn)

	ApiCfg := &handler.ApiConfig{
		FileserverHits: atomic.Int32{},
		Db:             dbQueries,
		Platform:       Platform,
		JwtSecret:      JwtSecret,
		PolkaKey:       PolkaKey,
	}

	mux := http.NewServeMux()
	fsHandler := ApiCfg.MiddlewareMetricsInc(http.StripPrefix("/app", http.FileServer(http.Dir(filepathRoot))))
	mux.Handle("/app/", fsHandler)

	mux.HandleFunc("GET /api/healthz", handler.HandlerReadiness)

	mux.HandleFunc("POST /api/polka/webhooks", ApiCfg.HandlerWebhook)

	mux.HandleFunc("POST /api/login", ApiCfg.HandlerLogin)
	mux.HandleFunc("POST /api/refresh", ApiCfg.HandlerRefresh)
	mux.HandleFunc("POST /api/revoke", ApiCfg.HandlerRevoke)

	mux.HandleFunc("POST /api/users", ApiCfg.HandlerUsersCreate)
	mux.HandleFunc("PUT /api/users", ApiCfg.HandlerUsersUpdate)

	mux.HandleFunc("POST /api/chirps", ApiCfg.HandlerChirpsCreate)
	mux.HandleFunc("GET /api/chirps", ApiCfg.HandlerChirpsRetrieve)
	mux.HandleFunc("GET /api/chirps/{chirpID}", ApiCfg.HandlerChirpsGet)
	mux.HandleFunc("DELETE /api/chirps/{chirpID}", ApiCfg.HandlerChirpsDelete)

	mux.HandleFunc("POST /admin/reset", ApiCfg.HandlerReset)
	mux.HandleFunc("GET /admin/metrics", ApiCfg.HandlerMetrics)

	srv := &http.Server{
		Addr:    ":" + port,
		Handler: mux,
	}

	log.Printf("Serving on port: %s\n", port)
	log.Fatal(srv.ListenAndServe())
}
