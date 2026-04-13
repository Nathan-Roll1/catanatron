/*
 * Grid search: sweep one parameter at a time across 100 values.
 * Each value: 1 seat with modified weight vs 3 baseline seats, CRN.
 * 10k games per value (504 per thread x 18 + baseline on same seeds).
 * Writes grid_log.jsonl for the dashboard.
 * Serves dashboard on localhost:8080.
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
#define NUM_W 11
#define GAMES_PER_VALUE 100008  /* per thread: 5556 games = ~100k per point */
#define GPT (GAMES_PER_VALUE / NT)
#define AB_DEPTH 2
#define NUM_POINTS 5

static const double ORIG[NUM_W] = {3e14,1e8,-1e8,1.,1e3,10.,1e2,1.,-5.,10.,10.1};
static const double DP[13] = {0,0,1./36,2./36,3./36,4./36,5./36,6./36,5./36,4./36,3./36,2./36,1./36};
static const double VB = 4.*(2.778/100.);
static const char *WN[NUM_W] = {"vps","prod","eprod","tiles","buildable","road","synergy","hand","discard","devs","army"};

static FILE *log_fp = NULL;
static volatile int running = 1;
static uint64_t run_seed_base;

/* Sweep range: +/- 0.5% */
static const double SWEEP_RANGE[NUM_W] = {
    0.005, 0.005, 0.005, 0.005, 0.005,
    0.005, 0.005, 0.005, 0.005, 0.005, 0.005
};

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

typedef struct {
    int tid, n;
    uint64_t seed_base;
    double test_w[NUM_W];
    int test_wins, base_wins, games;
} WorkerArgs;

static void play_one_batch(WorkerArgs *wa, const double w[NUM_W], int *wins) {
    RngState rng; SearchCtx ctx;
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    *wins = 0;
    for (int gi = 0; gi < wa->n; gi++) {
        int seat = gi % 4;
        GameWeights gw;
        for (int s = 0; s < 4; s++)
            memcpy(gw.W[s], (s == seat) ? w : ORIG, sizeof(double)*NUM_W);
        uint64_t seed = wa->seed_base + (uint64_t)wa->tid * 100000ULL + gi;
        rng_init(&rng, seed);
        CatanMap map; build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &rng);
        Game game; game_init_with_map(&game, &map, 4, colors, seed, 7, false, 10);
        game.eval_ctx = &gw;
        Action acts[MAX_ACTIONS];
        int n = generate_playable_actions(&game.state, acts, MAX_ACTIONS);
        while (game_winning_color(&game)==COLOR_NONE && game.state.num_turns<TURNS_LIMIT) {
            Action a; if (n==1) a=acts[0];
            else { Color cur=state_current_color(&game.state);
                ctx.depth_counter=0; Game cp; game_copy(&cp,&game); cp.eval_ctx=&gw;
                SearchResult sr=alphabeta_search(&ctx,&cp,acts,n,AB_DEPTH,-1e30,1e30,cur,game_eval);
                a=(sr.action.type||sr.action.color)?sr.action:acts[0]; }
            game_execute(&game,a,acts,&n);
        }
        Color w2=game_winning_color(&game);
        if (w2!=COLOR_NONE && game.state.color_to_index[(int)w2]==seat) (*wins)++;
    }
}

static void *worker(void *arg) {
    WorkerArgs *wa = (WorkerArgs*)arg;
    play_one_batch(wa, wa->test_w, &wa->test_wins);
    play_one_batch(wa, (const double*)ORIG, &wa->base_wins);
    wa->games = wa->n;
    return NULL;
}

typedef struct { double wr; double ci95; int tw; int bw; int n; } PointResult;

static PointResult run_point(const double test_w[NUM_W], uint64_t seed_base) {
    pthread_t threads[NT]; WorkerArgs wa[NT];
    for (int i = 0; i < NT; i++) {
        wa[i].tid = i; wa[i].n = GPT; wa[i].seed_base = seed_base;
        memcpy(wa[i].test_w, test_w, sizeof(double)*NUM_W);
    }
    for (int i = 0; i < NT; i++) pthread_create(&threads[i], NULL, worker, &wa[i]);
    for (int i = 0; i < NT; i++) pthread_join(threads[i], NULL);
    int tw = 0, bw = 0, tg = 0;
    for (int i = 0; i < NT; i++) { tw += wa[i].test_wins; bw += wa[i].base_wins; tg += wa[i].games; }
    double tp = (double)tw/tg, bp = (double)bw/tg;
    double diff = tp - bp;
    /* SE of difference of two proportions on same N (paired CRN reduces variance) */
    double se = sqrt((tp*(1-tp) + bp*(1-bp)) / tg);
    PointResult r;
    r.wr = 25.0 + diff * 100.0;
    r.ci95 = 1.96 * se * 100.0;
    r.tw = tw; r.bw = bw; r.n = tg;
    return r;
}

/* ---- HTTP ---- */

static char *dashboard_html = NULL;
static long dashboard_html_len = 0;

static void load_dashboard(const char *fname) {
    FILE *f = fopen(fname, "r");
    if (!f) return;
    fseek(f, 0, SEEK_END); dashboard_html_len = ftell(f); fseek(f, 0, SEEK_SET);
    dashboard_html = malloc(dashboard_html_len + 1);
    fread(dashboard_html, 1, dashboard_html_len, f);
    dashboard_html[dashboard_html_len] = 0;
    fclose(f);
}

static void *http_server(void *arg) {
    (void)arg;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1; setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr = {.sin_family=AF_INET,.sin_addr.s_addr=INADDR_ANY,.sin_port=htons(8080)};
    if (bind(fd,(struct sockaddr*)&addr,sizeof(addr))<0) { fprintf(stderr,"Port 8080 busy\n"); return NULL; }
    listen(fd, 16); fprintf(stderr, "Dashboard: http://localhost:8080\n");
    while (running) {
        struct sockaddr_in cl; socklen_t cll=sizeof(cl);
        int c = accept(fd,(struct sockaddr*)&cl,&cll); if (c<0) continue;
        char buf[4096]; int nr=read(c,buf,sizeof(buf)-1); if(nr<=0){close(c);continue;} buf[nr]=0;
        if (strstr(buf,"GET /data")) {
            FILE *f=fopen("grid_log.jsonl","r");
            if(f){fseek(f,0,SEEK_END);long sz=ftell(f);fseek(f,0,SEEK_SET);
                char *d=malloc(sz+1);fread(d,1,sz,f);d[sz]=0;fclose(f);
                char h[256];snprintf(h,256,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: %ld\r\n\r\n",sz);
                write(c,h,strlen(h));write(c,d,sz);free(d);}
        } else if (dashboard_html) {
            char h[256];snprintf(h,256,"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: %ld\r\n\r\n",dashboard_html_len);
            write(c,h,strlen(h));write(c,dashboard_html,dashboard_html_len);
        }
        close(c);
    }
    close(fd); return NULL;
}

int main(int argc, char **argv) {
    run_seed_base = (uint64_t)time(NULL);
    srand((unsigned)run_seed_base);
    signal(SIGPIPE, SIG_IGN);

    int param_idx = 0; /* default: sweep vps */
    if (argc > 1) param_idx = atoi(argv[1]);
    if (param_idx < 0 || param_idx >= NUM_W) param_idx = 0;

    RngState tmp; rng_init(&tmp, 0);
    CatanMap tmp_map; build_map(&tmp_map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &tmp);
    board_init_static_graph(&tmp_map);

    load_dashboard("grid_dashboard.html");
    pthread_t http_thread;
    pthread_create(&http_thread, NULL, http_server, NULL);

    log_fp = fopen("grid_log.jsonl", "w");

    double range = SWEEP_RANGE[param_idx];
    double lo = ORIG[param_idx] * (1.0 - range);
    double hi = ORIG[param_idx] * (1.0 + range);
    /* For negative params (eprod, discard), flip so lo < hi */
    if (lo > hi) { double t = lo; lo = hi; hi = t; }

    printf("=== Grid Search: %s ===\n", WN[param_idx]);
    printf("Range: %.4g to %.4g (orig=%.4g, +/-%.0f%%)\n", lo, hi, ORIG[param_idx], range*100);
    printf("%d points x %d games (CRN) x %d threads\n", NUM_POINTS, GAMES_PER_VALUE, NT);
    printf("Dashboard: http://localhost:8080\n\n");

    /* Write header to log */
    fprintf(log_fp, "{\"param\":\"%s\",\"param_idx\":%d,\"orig\":%.6g,\"lo\":%.6g,\"hi\":%.6g,\"points\":[]}\n",
            WN[param_idx], param_idx, ORIG[param_idx], lo, hi);
    fflush(log_fp);

    struct timespec T0, T1;
    clock_gettime(CLOCK_MONOTONIC, &T0);
    int total_games = 0;

    for (int pt = 0; pt < NUM_POINTS; pt++) {
        double val = lo + (hi - lo) * pt / (NUM_POINTS - 1);
        double test_w[NUM_W];
        memcpy(test_w, ORIG, sizeof(ORIG));
        test_w[param_idx] = val;

        uint64_t seed = run_seed_base * 1000003ULL + (uint64_t)pt * 131071ULL;
        PointResult pr = run_point(test_w, seed);
        total_games += GAMES_PER_VALUE * 2;

        double pct = (val - ORIG[param_idx]) / fabs(ORIG[param_idx]) * 100;

        clock_gettime(CLOCK_MONOTONIC, &T1);
        double elapsed = (T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)/1e9;
        double gps = total_games / elapsed;

        fprintf(log_fp, "{\"pt\":%d,\"val\":%.6g,\"pct\":%.4f,\"wr\":%.2f,\"ci95\":%.2f,\"tw\":%d,\"bw\":%d,\"n\":%d,\"games\":%d,\"elapsed\":%.1f,\"gps\":%.0f}\n",
                pt, val, pct, pr.wr, pr.ci95, pr.tw, pr.bw, pr.n, total_games, elapsed, gps);
        fflush(log_fp);

        printf("  [%d/%d] %s=%+.3f%%  wr=%.2f%% +/-%.2f%%  (test=%d base=%d n=%d)  [%dk, %.0fs, %.0f g/s]\n",
               pt+1, NUM_POINTS, WN[param_idx], pct, pr.wr, pr.ci95, pr.tw, pr.bw, pr.n,
               total_games/1000, elapsed, gps);
    }

    clock_gettime(CLOCK_MONOTONIC, &T1);
    double elapsed = (T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)/1e9;
    printf("\n=== Done: %dk games in %.0fs (%.0f g/s) ===\n", total_games/1000, elapsed, total_games/elapsed);

    if (log_fp) fclose(log_fp);
    printf("Dashboard at http://localhost:8080 (Ctrl-C to quit)\n");
    sleep(3600);
    return 0;
}
