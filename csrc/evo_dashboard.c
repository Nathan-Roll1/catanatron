/*
 * Parallel evolutionary optimizer with embedded HTTP dashboard.
 * 
 * EVERY game: 1 variant seat + 3 baseline seats (ORIG weights).
 * All variants play the SAME seeds each generation (CRN) for fair comparison.
 * Seeds randomized per run (time-based) and per generation.
 * 8 variants, 18 threads, selection every 2 gens (after 1k+ seats each).
 * Mutations: 0.1-1.1% perturbation on 1 random parameter.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"
#include "value.h"

#define NT 18
#define NUM_POP 8
#define NUM_W 11
#define GAMES_PER_VARIANT 504       /* per gen: 504 games x 4 seats / 4 = 504 seat-appearances */
#define GAMES_PER_GEN (GAMES_PER_VARIANT) /* each variant plays this many games */
#define GPT_PER_VARIANT (GAMES_PER_VARIANT / NT) /* 28 games per thread per variant */
#define AB_DEPTH 2
#define RUN_SECONDS 60
#define SELECT_EVERY 2

static const double ORIG[NUM_W] = {3e14,1e8,-1e8,1.,1e3,10.,1e2,1.,-5.,10.,10.1};
static const double DP[13] = {0,0,1./36,2./36,3./36,4./36,5./36,6./36,5./36,4./36,3./36,2./36,1./36};
static const double VB = 4.*(2.778/100.);
static const char *WN[NUM_W] = {"vps","prod","eprod","tiles","buildable","road","synergy","hand","discard","devs","army"};

typedef struct {
    int id, parent_id, generation_born;
    double w[NUM_W];
    int gen_wins, gen_seats;     /* this generation's stats */
    int total_wins, total_seats; /* lifetime stats */
    double wr;                   /* current win rate vs baseline */
} Variant;

static Variant population[NUM_POP];
static int next_variant_id = NUM_POP;
static FILE *log_fp = NULL;
static volatile int running = 1;
static uint64_t run_seed_base;

/* ---- Eval ---- */

typedef struct { double W[4][NUM_W]; } GameWeights;

static double eval_w(Game *g, Color c, const double *W) {
    State *s=&g->state;int idx=s->color_to_index[(int)c];
    Board *b=&s->board;CatanMap *map=b->map;Coordinate robber=b->robber_coordinate;
    double rp[5]={0};
    for(int si=0;si<s->settlement_count[idx];si++){int node=s->settlements[idx][si];
        if(node<0||node>=NUM_NODES)continue;
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
            if(t2<0||t2>=NUM_LAND_TILES)continue;
            LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
            if(coord_eq(map->land_tile_coords[t2],robber))continue;rp[(int)t->resource]+=DP[t->number];}}
    for(int ci=0;ci<s->city_count[idx];ci++){int node=s->cities[idx][ci];
        if(node<0||node>=NUM_NODES)continue;
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
            if(t2<0||t2>=NUM_LAND_TILES)continue;
            LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
            if(coord_eq(map->land_tile_coords[t2],robber))continue;rp[(int)t->resource]+=2.*DP[t->number];}}
    double tp=0;int var=0;for(int r=0;r<5;r++){tp+=rp[r];if(rp[r]>0)var++;}
    double prod=tp+var*VB;
    Color enemy=COLOR_NONE;for(int i=0;i<s->num_players;i++)if(s->colors[i]!=c){enemy=s->colors[i];break;}
    double ep=0;
    if(enemy!=COLOR_NONE){int ei=s->color_to_index[(int)enemy];
        for(int si=0;si<s->settlement_count[ei];si++){int node=s->settlements[ei][si];
            if(node<0||node>=NUM_NODES)continue;
            for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
                if(t2<0||t2>=NUM_LAND_TILES)continue;
                LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
                if(coord_eq(map->land_tile_coords[t2],robber))continue;ep+=DP[t->number];}}
        for(int ci=0;ci<s->city_count[ei];ci++){int node=s->cities[ei][ci];
            if(node<0||node>=NUM_NODES)continue;
            for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
                if(t2<0||t2>=NUM_LAND_TILES)continue;
                LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
                if(coord_eq(map->land_tile_coords[t2],robber))continue;ep+=2.*DP[t->number];}}}
    int *ps=s->player_state[idx];
    int wh=ps[PS_WHEAT_IN_HAND],or2=ps[PS_ORE_IN_HAND],sh=ps[PS_SHEEP_IN_HAND],
        br=ps[PS_BRICK_IN_HAND],wo=ps[PS_WOOD_IN_HAND];
    double dc=(fmax(2-wh,0)+fmax(3-or2,0))/5.;double ds=(fmax(1-wh,0)+fmax(1-sh,0)+fmax(1-br,0)+fmax(1-wo,0))/4.;
    double syn=(2-dc-ds)/2.;int nih=wo+br+sh+wh+or2;
    bool ts[NUM_LAND_TILES]={0};int nt=0;
    for(int si=0;si<s->settlement_count[idx];si++){int node=s->settlements[idx][si];
        if(node<0||node>=NUM_NODES)continue;
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];if(t2>=0&&t2<NUM_LAND_TILES&&!ts[t2]){ts[t2]=1;nt++;}}}
    for(int ci=0;ci<s->city_count[idx];ci++){int node=s->cities[idx][ci];
        if(node<0||node>=NUM_NODES)continue;
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];if(t2>=0&&t2<NUM_LAND_TILES&&!ts[t2]){ts[t2]=1;nt++;}}}
    uint64_t reach[2]={0,0};
    for(int i=0;i<s->board.cc_count[(int)c];i++)bs_or(reach,reach,s->board.cc_sets[(int)c][i]);
    uint64_t avail[2];bs_and(avail,reach,s->board.buildable);
    int nb=__builtin_popcountll(avail[0])+__builtin_popcountll(avail[1]);
    double lrf=(nb==0)?W[5]:0.1;
    int nd=ps[PS_KNIGHT_IN_HAND]+ps[PS_YEAR_OF_PLENTY_IN_HAND]+ps[PS_MONOPOLY_IN_HAND]
          +ps[PS_ROAD_BUILDING_IN_HAND]+ps[PS_VICTORY_POINT_IN_HAND];
    return ps[PS_VICTORY_POINTS]*W[0]+prod*W[1]+ep*W[2]+nt*W[3]+nb*W[4]
          +ps[PS_LONGEST_ROAD_LENGTH]*lrf+syn*W[6]+nih*W[7]+(nih>7?W[8]:0)+nd*W[9]+ps[PS_PLAYED_KNIGHT]*W[10];
}

static double game_eval(Game *g, Color c) {
    int idx = g->state.color_to_index[(int)c];
    if (idx < 0 || idx >= 4 || !g->eval_ctx) return base_value_fn(g, c);
    return eval_w(g, c, ((const GameWeights *)g->eval_ctx)->W[idx]);
}

/* ---- Worker: play variant in seat (gi%4) vs 3 baseline, on shared seeds ---- */

typedef struct {
    int tid, n;
    uint64_t seed_base;
    double variant_w[NUM_W];
    int wins, games;
    /* Also run baseline on same seeds for CRN */
    int base_wins;
} WorkerArgs;

static void play_batch(WorkerArgs *wa, const double test_w[NUM_W], int *out_wins) {
    RngState rng; SearchCtx ctx;
    Color colors[4] = {COLOR_RED, COLOR_BLUE, COLOR_ORANGE, COLOR_WHITE};
    *out_wins = 0;

    for (int gi = 0; gi < wa->n; gi++) {
        int test_seat = gi % 4;
        GameWeights gw;
        for (int s = 0; s < 4; s++)
            memcpy(gw.W[s], (s == test_seat) ? test_w : ORIG, sizeof(double) * NUM_W);

        uint64_t seed = wa->seed_base + (uint64_t)wa->tid * 100000ULL + gi;
        rng_init(&rng, seed);
        CatanMap map; build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &rng);
        Game game; game_init_with_map(&game, &map, 4, colors, seed, 7, false, 10);
        game.eval_ctx = &gw;

        Action acts[MAX_ACTIONS];
        int n = generate_playable_actions(&game.state, acts, MAX_ACTIONS);
        while (game_winning_color(&game) == COLOR_NONE && game.state.num_turns < TURNS_LIMIT) {
            Action a;
            if (n == 1) { a = acts[0]; }
            else {
                Color cur = state_current_color(&game.state);
                ctx.depth_counter = 0;
                Game cp; game_copy(&cp, &game); cp.eval_ctx = &gw;
                SearchResult sr = alphabeta_search(&ctx, &cp, acts, n, AB_DEPTH, -1e30, 1e30, cur, game_eval);
                a = (sr.action.type != 0 || sr.action.color != 0) ? sr.action : acts[0];
            }
            game_execute(&game, a, acts, &n);
        }
        Color w = game_winning_color(&game);
        if (w != COLOR_NONE && game.state.color_to_index[(int)w] == test_seat)
            (*out_wins)++;
    }
}

static void *worker(void *arg) {
    WorkerArgs *wa = (WorkerArgs *)arg;
    /* Play variant vs baseline on these seeds */
    play_batch(wa, wa->variant_w, &wa->wins);
    /* Play baseline vs baseline on SAME seeds for CRN comparison */
    play_batch(wa, (const double *)ORIG, &wa->base_wins);
    wa->games = wa->n;
    return NULL;
}

/* ---- Mutation ---- */

static void mutate(Variant *child, const Variant *parent, int gen) {
    *child = *parent;
    child->id = next_variant_id++;
    child->parent_id = parent->id;
    child->generation_born = gen;
    child->gen_wins = 0; child->gen_seats = 0;
    child->total_wins = 0; child->total_seats = 0;
    child->wr = 25.0;
    int p = rand() % NUM_W;
    double pct = (0.01 + (rand() % 10) / 100.0) / 100.0; /* 0.01% to 0.11% */
    if (rand() % 2) pct = -pct;
    child->w[p] *= (1.0 + pct);
}

/* ---- Log ---- */

static void log_gen(int gen, double elapsed, int total_games, double gps) {
    if (!log_fp) return;
    fprintf(log_fp, "{\"gen\":%d,\"elapsed\":%.1f,\"total_games\":%d,\"gps\":%.0f,\"val_wr\":%.2f,\"variants\":[",
            gen, elapsed, total_games, gps,
            population[0].wr); /* champion is always slot 0 after sorting */
    for (int i = 0; i < NUM_POP; i++) {
        Variant *v = &population[i];
        fprintf(log_fp, "%s{\"id\":%d,\"parent\":%d,\"born\":%d,\"wr\":%.2f,\"wins\":%d,\"seats\":%d,\"w\":[",
                i ? "," : "", v->id, v->parent_id, v->generation_born,
                v->wr, v->total_wins, v->total_seats);
        for (int j = 0; j < NUM_W; j++)
            fprintf(log_fp, "%s%.6g", j ? "," : "", v->w[j]);
        fprintf(log_fp, "]}");
    }
    fprintf(log_fp, "]}\n");
    fflush(log_fp);
}

/* ---- HTTP server ---- */

static char *dashboard_html = NULL;
static long dashboard_html_len = 0;

static void load_dashboard(void) {
    FILE *f = fopen("dashboard.html", "r");
    if (!f) { fprintf(stderr, "Warning: dashboard.html not found\n"); return; }
    fseek(f, 0, SEEK_END); dashboard_html_len = ftell(f); fseek(f, 0, SEEK_SET);
    dashboard_html = malloc(dashboard_html_len + 1);
    fread(dashboard_html, 1, dashboard_html_len, f);
    dashboard_html[dashboard_html_len] = 0;
    fclose(f);
}

static void *http_server(void *arg) {
    (void)arg;
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1; setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_addr.s_addr = INADDR_ANY, .sin_port = htons(8080)};
    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "Cannot bind to port 8080\n"); return NULL;
    }
    listen(server_fd, 16);
    fprintf(stderr, "Dashboard: http://localhost:8080\n");
    while (running) {
        struct sockaddr_in client; socklen_t cl = sizeof(client);
        int cfd = accept(server_fd, (struct sockaddr *)&client, &cl);
        if (cfd < 0) continue;
        char buf[4096]; int nr = read(cfd, buf, sizeof(buf) - 1);
        if (nr <= 0) { close(cfd); continue; }
        buf[nr] = 0;
        if (strstr(buf, "GET /data")) {
            FILE *f = fopen("evo_log.jsonl", "r");
            if (f) {
                fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
                char *data = malloc(sz + 1); fread(data, 1, sz, f); data[sz] = 0; fclose(f);
                char hdr[256];
                snprintf(hdr, sizeof(hdr), "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: %ld\r\n\r\n", sz);
                write(cfd, hdr, strlen(hdr)); write(cfd, data, sz); free(data);
            }
        } else if (dashboard_html) {
            char hdr[256];
            snprintf(hdr, sizeof(hdr), "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: %ld\r\n\r\n", dashboard_html_len);
            write(cfd, hdr, strlen(hdr)); write(cfd, dashboard_html, dashboard_html_len);
        }
        close(cfd);
    }
    close(server_fd);
    return NULL;
}

/* ---- Main ---- */

int main(void) {
    run_seed_base = (uint64_t)time(NULL);
    srand((unsigned)run_seed_base);
    signal(SIGPIPE, SIG_IGN);

    RngState tmp; rng_init(&tmp, 0);
    CatanMap tmp_map; build_map(&tmp_map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &tmp);
    board_init_static_graph(&tmp_map);

    load_dashboard();
    pthread_t http_thread;
    pthread_create(&http_thread, NULL, http_server, NULL);

    for (int i = 0; i < NUM_POP; i++) {
        memcpy(population[i].w, ORIG, sizeof(ORIG));
        population[i].id = i; population[i].parent_id = -1;
        population[i].generation_born = 0; population[i].wr = 25.0;
        population[i].gen_wins = population[i].gen_seats = 0;
        population[i].total_wins = population[i].total_seats = 0;
    }
    for (int i = 1; i < NUM_POP; i++) mutate(&population[i], &population[0], 0);

    log_fp = fopen("evo_log.jsonl", "w");

    struct timespec T0, T1;
    clock_gettime(CLOCK_MONOTONIC, &T0);
    int total_games = 0, gen = 0;

    printf("=== Evolutionary Optimizer (all games vs baseline) ===\n");
    printf("Pop: %d | Games/variant/gen: %d | Threads: %d | AB:%d\n", NUM_POP, GAMES_PER_VARIANT, NT, AB_DEPTH);
    printf("CRN: every variant plays same seeds, compared to baseline on same seeds\n");
    printf("Run seed: %llu | Run time: %ds\n", (unsigned long long)run_seed_base, RUN_SECONDS);
    printf("Dashboard: http://localhost:8080\n\n");

    while (1) {
        clock_gettime(CLOCK_MONOTONIC, &T1);
        double elapsed = (T1.tv_sec - T0.tv_sec) + (T1.tv_nsec - T0.tv_nsec) / 1e9;
        if (elapsed >= RUN_SECONDS) break;
        gen++;

        uint64_t gen_seed = run_seed_base * 1000003ULL + (uint64_t)gen * 131071ULL;

        /* Evaluate each variant: play vs baseline on same seeds */
        for (int vi = 0; vi < NUM_POP; vi++) {
            pthread_t threads[NT]; WorkerArgs wa[NT];
            for (int i = 0; i < NT; i++) {
                wa[i].tid = i; wa[i].n = GPT_PER_VARIANT;
                wa[i].seed_base = gen_seed; /* SAME seeds for all variants */
                memcpy(wa[i].variant_w, population[vi].w, sizeof(double) * NUM_W);
            }
            for (int i = 0; i < NT; i++) pthread_create(&threads[i], NULL, worker, &wa[i]);
            for (int i = 0; i < NT; i++) pthread_join(threads[i], NULL);

            int v_wins = 0, b_wins = 0, games = 0;
            for (int i = 0; i < NT; i++) {
                v_wins += wa[i].wins;
                b_wins += wa[i].base_wins;
                games += wa[i].games;
            }
            total_games += games * 2; /* variant + baseline runs */

            population[vi].gen_wins = v_wins;
            population[vi].gen_seats = games;
            population[vi].total_wins += v_wins;
            population[vi].total_seats += games;

            /* CRN win rate: 25% + (variant_wr - baseline_wr) */
            double vwr = 100.0 * v_wins / games;
            double bwr = 100.0 * b_wins / games;
            population[vi].wr = 25.0 + (vwr - bwr);
        }

        clock_gettime(CLOCK_MONOTONIC, &T1);
        elapsed = (T1.tv_sec - T0.tv_sec) + (T1.tv_nsec - T0.tv_nsec) / 1e9;
        double gps = total_games / elapsed;
        log_gen(gen, elapsed, total_games, gps);

        if (gen % 5 == 0) {
            printf("gen %3d: ", gen);
            for (int p = 0; p < NUM_POP; p++)
                printf("V%d=%.1f%% ", population[p].id, population[p].wr);
            printf("[%dk, %.0fs, %.0f g/s]\n", total_games / 1000, elapsed, gps);
        }

        /* Selection */
        if (gen % SELECT_EVERY == 0) {
            int rank[NUM_POP]; for (int i = 0; i < NUM_POP; i++) rank[i] = i;
            for (int i = 0; i < NUM_POP - 1; i++)
                for (int j = i + 1; j < NUM_POP; j++)
                    if (population[rank[j]].wr > population[rank[i]].wr)
                        { int t = rank[i]; rank[i] = rank[j]; rank[j] = t; }

            Variant new_pop[NUM_POP];
            for (int i = 0; i < NUM_POP / 2; i++) new_pop[i] = population[rank[i]];
            for (int i = NUM_POP / 2; i < NUM_POP; i++)
                mutate(&new_pop[i], &new_pop[rand() % (NUM_POP / 2)], gen);
            memcpy(population, new_pop, sizeof(population));
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &T1);
    double elapsed = (T1.tv_sec - T0.tv_sec) + (T1.tv_nsec - T0.tv_nsec) / 1e9;
    printf("\n=== Done: %d gens, %dk games, %.0fs, %.0f g/s ===\n",
           gen, total_games / 1000, elapsed, total_games / elapsed);

    int best = 0;
    for (int i = 1; i < NUM_POP; i++)
        if (population[i].wr > population[best].wr) best = i;
    printf("Champion V%d (%.1f%% vs 25%% baseline):\n", population[best].id, population[best].wr);
    for (int i = 0; i < NUM_W; i++) {
        double pct = (population[best].w[i] - ORIG[i]) / fabs(ORIG[i]) * 100;
        if (fabs(pct) > 0.01) printf("  %-12s %+.3f%%\n", WN[i], pct);
    }

    if (log_fp) fclose(log_fp);
    running = 0;
    printf("\nDashboard at http://localhost:8080 (Ctrl-C to quit)\n");
    sleep(3600);
    return 0;
}
