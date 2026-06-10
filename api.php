<?php
/**
 * Essay Checker API
 * Mode 1: TF-IDF Enhanced — blended pre-built IDF + Jaccard + BM25
 * Mode 2: IndoBERT — HuggingFace Inference API (indobenchmark/indobert-base-p2)
 */

header('Content-Type: application/json; charset=UTF-8');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type, Authorization');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') { http_response_code(200); exit; }
if ($_SERVER['REQUEST_METHOD'] !== 'POST')    { send_error(405, 'Method not allowed. Gunakan POST.'); }

$action = $_GET['action'] ?? 'analyze';
switch ($action) {
    case 'analyze':    handle_analyze();   break;
    case 'similarity': handle_similarity_only(); break;
    case 'embedding':  handle_embedding(); break;
    default:           send_error(400, 'Action tidak dikenal.');
}

// ================================================================
// HANDLER UTAMA
// ================================================================
function handle_analyze() {
    $input         = get_input();
    $kunci_raw     = $input['kunci'] ?? [];
    $kunci_list    = is_string($kunci_raw) ? json_decode($kunci_raw, true) : $kunci_raw;
    $jawaban       = trim($input['jawaban'] ?? '');
    $threshold     = floatval($input['threshold'] ?? 0.6);
    $w_semantic    = floatval($input['w_semantic'] ?? 60) / 100;
    $w_kelengkapan = 1.0 - $w_semantic;

    if (empty($kunci_list)) send_error(400, 'Kunci jawaban tidak boleh kosong.');
    if (empty($jawaban) && isset($_FILES['file'])) $jawaban = extract_text_from_file($_FILES['file']);
    if (empty($jawaban))    send_error(400, 'Jawaban mahasiswa tidak boleh kosong.');

    // PRE-PROCESSING
    $jawaban_bersih = preprocess_text($jawaban);
    $kunci_gabungan = implode('. ', $kunci_list);
    $kunci_bersih   = preprocess_text($kunci_gabungan);
    $contoh_stem    = get_stem_examples($jawaban . ' ' . $kunci_gabungan, 5);

    // TF-IDF (dipakai untuk BM25 & keyword coverage di kedua mode)
    $semua_dok = [$jawaban_bersih, $kunci_bersih];
    foreach ($kunci_list as $p) $semua_dok[] = preprocess_text($p);
    $vocab    = build_vocab($semua_dok);
    $idf      = compute_idf_blended($semua_dok, $vocab);
    $top_tfidf = get_top_tfidf($jawaban_bersih, $vocab, $idf, 8);

    $avg_len = array_sum(array_map(
        fn($d) => count(array_filter(explode(' ', $d))),
        $semua_dok
    )) / count($semua_dok);

    // ── MODE DETECTION ──────────────────────────────────────────────
    $mode     = 'tfidf';
    $hf_error = null;

    $emb_jawaban = null;
    $emb_kunci   = null;
    $emb_per_poin = [];

    $all_texts   = array_merge([$jawaban, $kunci_gabungan], $kunci_list);
    $local_result = get_local_embeddings($all_texts);
    if (is_array($local_result) && count($local_result) >= 2) {
        $mode         = 'indobert';
        $emb_jawaban  = $local_result[0];
        $emb_kunci    = $local_result[1];
        $emb_per_poin = array_slice($local_result, 2);
    } else {
        $hf_error = is_string($local_result) ? $local_result : 'IndoBERT server tidak aktif, fallback TF-IDF.';
    }

    if ($mode === 'tfidf') {
        $emb_jawaban = compute_tfidf($jawaban_bersih, $vocab, $idf);
        $emb_kunci   = compute_tfidf($kunci_bersih,   $vocab, $idf);
        foreach ($kunci_list as $p) {
            $emb_per_poin[] = compute_tfidf(preprocess_text($p), $vocab, $idf);
        }
    }

    // COSINE SIMILARITY keseluruhan
    $cosine_sim  = cosine_similarity($emb_jawaban, $emb_kunci);
    $cosine_dist = 1.0 - $cosine_sim;

    // JACCARD SIMILARITY (tambahan sinyal)
    $jaccard = jaccard_similarity($jawaban_bersih, $kunci_bersih);

    // Gabungkan cosine + jaccard untuk skor semantic (bobot: 85% cosine, 15% jaccard)
    $semantic_blended = $cosine_sim * 0.85 + $jaccard * 0.15;

    // SIMILARITY PER POIN
    $poin_results          = [];
    $poin_terpenuhi_soft   = 0.0;
    $poin_terpenuhi_binary = 0;
    $raw_bm25_scores       = [];

    foreach ($kunci_list as $i => $poin) {
        $poin_bersih = preprocess_text($poin);
        $emb_poin    = $emb_per_poin[$i] ?? ($mode === 'tfidf'
            ? compute_tfidf($poin_bersih, $vocab, $idf)
            : $emb_kunci);

        $sim_cosine  = cosine_similarity($emb_jawaban, $emb_poin);
        $sim_jaccard = jaccard_similarity($jawaban_bersih, $poin_bersih);
        $bm25_raw    = compute_bm25($poin_bersih, $jawaban_bersih, $vocab, $idf, 1.5, 0.75, $avg_len);
        $raw_bm25_scores[] = $bm25_raw;

        // Blended per-poin similarity: cosine 70% + jaccard 30% (mode tfidf)
        // IndoBERT mode: cosine 85% + jaccard 15% (cosine lebih dipercaya)
        $blend_ratio = ($mode === 'indobert') ? [0.85, 0.15] : [0.70, 0.30];
        $sim_poin    = $sim_cosine * $blend_ratio[0] + $sim_jaccard * $blend_ratio[1];

        $partial  = soft_completeness($sim_poin, $threshold);
        $coverage = keyword_coverage($jawaban_bersih, $poin_bersih, $vocab, $idf, 5);

        $poin_terpenuhi_soft   += $partial;
        if ($sim_poin >= $threshold) $poin_terpenuhi_binary++;

        $poin_results[] = [
            'no'               => $i + 1,
            'poin'             => $poin,
            'similarity'       => round($sim_poin, 4),
            'sim_cosine'       => round($sim_cosine, 4),
            'sim_jaccard'      => round($sim_jaccard, 4),
            'terpenuhi'        => $sim_poin >= $threshold,
            'label'            => get_label($sim_poin),
            'partial_credit'   => round($partial, 4),
            'keyword_coverage' => $coverage['ratio'],
            'kata_kunci_poin'  => array_merge($coverage['matched'], $coverage['missing']),
            'kata_hadir'       => $coverage['matched'],
            'kata_hilang'      => $coverage['missing'],
            'bm25_raw'         => round($bm25_raw, 4),
            'bm25_norm'        => 0.0,
        ];
    }

    // Normalisasi BM25
    $max_bm25 = count($raw_bm25_scores) > 0 ? max($raw_bm25_scores) : 0.0;
    foreach ($poin_results as &$r) {
        $r['bm25_norm'] = ($max_bm25 > 0.0) ? round($r['bm25_raw'] / $max_bm25, 4) : 0.0;
    }
    unset($r);

    // SKOR AKHIR
    $rasio_kelengkapan        = count($kunci_list) > 0 ? $poin_terpenuhi_soft / count($kunci_list) : 0;
    $rasio_kelengkapan_binary = count($kunci_list) > 0 ? $poin_terpenuhi_binary / count($kunci_list) : 0;

    // BM25 avg sebagai booster kecil pada kelengkapan
    $avg_bm25_norm = count($poin_results) > 0
        ? array_sum(array_column($poin_results, 'bm25_norm')) / count($poin_results) : 0.0;
    $rasio_kelengkapan_boosted = min(1.0, $rasio_kelengkapan * 0.85 + $avg_bm25_norm * 0.15);

    $skor_semantic   = $semantic_blended * 100;
    $skor_kelengkapan = $rasio_kelengkapan_boosted * 100;
    $skor_akhir      = ($skor_semantic * $w_semantic) + ($skor_kelengkapan * $w_kelengkapan);

    $rata_coverage = count($poin_results) > 0
        ? array_sum(array_column($poin_results, 'keyword_coverage')) / count($poin_results) : 0.0;

    $tokens_jawaban = explode(' ', $jawaban_bersih);
    $tokens_kunci   = explode(' ', $kunci_bersih);

    $dim = count($emb_jawaban);

    echo json_encode([
        'status' => 'success',
        'config' => [
            'model'         => $mode === 'indobert'
                ? 'IndoBERT Base P2 (indobenchmark/indobert-base-p2) Lokal + BM25 + Jaccard'
                : 'TF-IDF Enhanced (Pre-built IDF) + BM25 + Jaccard',
            'mode'          => $mode,
            'threshold'     => $threshold,
            'w_semantic'    => $w_semantic,
            'w_kelengkapan' => $w_kelengkapan,
            'hf_error'      => $hf_error,
        ],
        'preprocessing' => [
            'jawaban_original' => $jawaban,
            'jawaban_bersih'   => $jawaban_bersih,
            'kunci_original'   => $kunci_gabungan,
            'kunci_bersih'     => $kunci_bersih,
            'panjang_jawaban'  => count(array_filter($tokens_jawaban)),
            'panjang_kunci'    => count(array_filter($tokens_kunci)),
            'ukuran_vocab'     => count($vocab),
            'stemming_aktif'   => true,
            'contoh_stem'      => $contoh_stem,
        ],
        'embedding' => [
            'metode'       => $mode === 'indobert'
                ? 'IndoBERT Mean Pooling (last_hidden_state, 768 dim)'
                : 'TF-IDF Enhanced (Pre-built IDF corpus)',
            'dimensi'      => $dim,
            'norm_jawaban' => round(vektor_norm($emb_jawaban), 6),
            'norm_kunci'   => round(vektor_norm($emb_kunci),   6),
            'dot_product'  => round(dot_product($emb_jawaban, $emb_kunci), 6),
            'top_tfidf'    => $mode === 'tfidf' ? $top_tfidf : [],
        ],
        'similarity' => [
            'cosine_similarity' => round($cosine_sim, 6),
            'cosine_distance'   => round($cosine_dist, 6),
            'jaccard_similarity'=> round($jaccard, 6),
            'blended_similarity'=> round($semantic_blended, 6),
            'label'             => get_label($semantic_blended),
            'threshold'         => $threshold,
        ],
        'poin_detail' => $poin_results,
        'scoring' => [
            'poin_terpenuhi'          => round($poin_terpenuhi_soft, 4),
            'poin_terpenuhi_soft'     => round($poin_terpenuhi_soft, 4),
            'poin_terpenuhi_binary'   => $poin_terpenuhi_binary,
            'total_poin'              => count($kunci_list),
            'rasio_kelengkapan'       => round($rasio_kelengkapan, 4),
            'skor_semantic'           => round($skor_semantic, 2),
            'skor_kelengkapan'        => round($skor_kelengkapan, 2),
            'skor_kelengkapan_soft'   => round($rasio_kelengkapan * 100, 2),
            'skor_kelengkapan_binary' => round($rasio_kelengkapan_binary * 100, 2),
            'skor_akhir'              => round($skor_akhir, 2),
            'rata_coverage_poin'      => round($rata_coverage, 4),
            'skor_keyword_coverage'   => round($rata_coverage * 100, 2),
            'avg_bm25_norm'           => round($avg_bm25_norm, 4),
            'formula'                 => sprintf(
                '(%.1f × %.2f) + (%.1f × %.2f) = %.2f',
                $skor_semantic, $w_semantic,
                $skor_kelengkapan, $w_kelengkapan,
                $skor_akhir
            ),
        ],
    ], JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE);
}

function handle_similarity_only() {
    $input = get_input();
    $teks1 = trim($input['teks1'] ?? '');
    $teks2 = trim($input['teks2'] ?? '');
    if (empty($teks1) || empty($teks2)) send_error(400, 'teks1 dan teks2 tidak boleh kosong.');

    $b1    = preprocess_text($teks1);
    $b2    = preprocess_text($teks2);
    $vocab = build_vocab([$b1, $b2]);
    $idf   = compute_idf_blended([$b1, $b2], $vocab);
    $e1    = compute_tfidf($b1, $vocab, $idf);
    $e2    = compute_tfidf($b2, $vocab, $idf);
    $sim   = cosine_similarity($e1, $e2);
    $jac   = jaccard_similarity($b1, $b2);

    echo json_encode([
        'status'            => 'success',
        'teks1'             => $teks1,
        'teks2'             => $teks2,
        'cosine_similarity' => round($sim, 6),
        'jaccard_similarity'=> round($jac, 6),
        'blended_similarity'=> round($sim * 0.8 + $jac * 0.2, 6),
        'cosine_distance'   => round(1.0 - $sim, 6),
        'label'             => get_label($sim * 0.8 + $jac * 0.2),
        'dot_product'       => round(dot_product($e1, $e2), 6),
        'norm_a'            => round(vektor_norm($e1), 6),
        'norm_b'            => round(vektor_norm($e2), 6),
    ], JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE);
}

function handle_embedding() {
    $input = get_input();
    $teks  = trim($input['teks'] ?? '');
    if (empty($teks)) send_error(400, 'Teks tidak boleh kosong.');
    $bersih = preprocess_text($teks);
    $vocab  = build_vocab([$bersih]);
    $idf    = compute_idf_blended([$bersih], $vocab);
    $emb    = compute_tfidf($bersih, $vocab, $idf);
    echo json_encode([
        'status'  => 'success',
        'teks'    => $teks,
        'bersih'  => $bersih,
        'dimensi' => count($vocab),
        'norm'    => round(vektor_norm($emb), 6),
        'top_10'  => get_top_tfidf($bersih, $vocab, $idf, 10),
    ], JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE);
}

// ================================================================
// INDOBERT LOCAL SERVER INTEGRATION
// Requires: indobert_server.py running on localhost:5001
// ================================================================

function get_local_embeddings(array $texts): array|string {
    if (!function_exists('curl_init')) return 'curl tidak tersedia.';

    $ch = curl_init('http://127.0.0.1:5001/embed');
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_POST           => true,
        CURLOPT_TIMEOUT        => 120,
        CURLOPT_HTTPHEADER     => ['Content-Type: application/json'],
        CURLOPT_POSTFIELDS     => json_encode(['texts' => $texts]),
    ]);

    $raw  = curl_exec($ch);
    $code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $err  = curl_error($ch);
    curl_close($ch);

    if ($raw === false || $code === 0) return 'IndoBERT server tidak aktif (jalankan indobert_server.py).';
    if ($code !== 200) return "IndoBERT server error HTTP $code.";

    $data = json_decode($raw, true);
    if (!is_array($data) || !isset($data['embeddings'])) return 'Response tidak valid dari IndoBERT server.';

    $embs = $data['embeddings'];
    return count($embs) === count($texts) ? $embs : 'Jumlah embedding tidak sesuai.';
}

// ================================================================
// PRE-BUILT IDF (approximate from large Indonesian corpus)
// Kunci: bentuk STEM setelah preprocess_text(), bukan kata asli.
// ================================================================

function get_pretrained_idf(): array {
    return [
        // ── UMUM AKADEMIK (2.0–3.5) ──────────────────────────────
        'sistem'          => 2.1,  'proses'       => 2.2,
        'aspek'           => 2.3,  'faktor'       => 2.4,
        'komponen'        => 2.5,  'model'        => 2.6,
        'analisis'        => 2.7,  'konsep'       => 2.8,
        'teor'            => 2.9,  // teori
        'struktur'        => 3.0,  'output'       => 2.0,
        'input'           => 2.0,  'sampel'       => 2.8,
        'tode'            => 2.8,  // metode
        'ilai'            => 2.5,  // penilaian
        'ukur'            => 2.6,  // pengukuran
        'nila'            => 2.5,  // nilai
        'tuju'            => 2.5,  // tujuan
        'prinsip'         => 3.2,  'teknik'       => 3.0,
        'belajar'         => 2.5,  'strateg'      => 3.3, // strategi
        'dekat'           => 2.5,  // pendekatan
        'data'            => 2.8,  'hasil'        => 2.2,
        'hubungan'        => 3.2,  'dampak'       => 3.5,
        'pengaruh'        => 3.5,  'manfaat'      => 3.2,
        'tingkat'         => 2.8,  'kondisi'      => 3.0,
        'tipe'            => 2.8,  'kategori'     => 3.2,
        'kesimpul'        => 3.5,  // kesimpulan
        'tesis'           => 4.8,  'hipotes'      => 5.0, // hipotesis (alt stem)

        // ── AKADEMIK FORMAL (3.5–5.5) ────────────────────────────
        'algoritma'       => 4.5,  'variabel'     => 3.8,
        'hipotesis'       => 5.0,  'korelas'      => 4.2, // korelasi
        'validitas'       => 4.8,  'reliabilitas' => 5.0,
        'distribus'       => 4.0,  // distribusi
        'populas'         => 3.8,  // populasi
        'implementas'     => 3.7,  // implementasi
        'evaluas'         => 3.6,  // evaluasi
        'karakteristik'   => 3.8,  'parameter'    => 4.0,
        'kompetens'       => 4.0,  // kompetensi
        'terampil'        => 3.9,  // keterampilan
        'etahu'           => 3.5,  // pengetahuan (peng → etahu)
        'aham'            => 3.5,  // pemahaman
        'identifikas'     => 4.0,  'observas'     => 4.0, // observasi
        'eksperimen'      => 4.5,  'interprets'   => 4.2,
        'klasifiks'       => 4.5,  'cahaya'       => 3.5,
        'karbon'          => 5.5,  'energi'       => 4.5,
        'energ'           => 4.5,  // energi (stripped i)

        // ── BIOLOGI (5.5–8.5) ────────────────────────────────────
        'fotosintesis'    => 7.5,  'ekosistem'    => 6.5,
        'genetika'        => 6.8,  'organisme'    => 5.5,
        'enzim'           => 6.5,  'protein'      => 5.2,
        'hormon'          => 6.0,  'neuron'       => 6.5,
        'osmosis'         => 7.2,  'tabolisme'    => 6.5, // metabolisme
        'respiras'        => 6.2,  // respirasi
        'evolus'          => 6.2,  // evolusi
        'transpiras'      => 7.5,  // transpirasi
        'sel'             => 4.5,  'klorofil'     => 8.0,
        'glukosa'         => 7.8,  'oksigen'      => 6.2,
        'oksida'          => 6.0,  // dioksida
        'kromosom'        => 7.5,  'gen'          => 5.5,
        'mutasi'          => 6.5,  'sitoplasma'   => 7.8,
        'membran'         => 5.5,  'nukleus'      => 7.0,
        'mitokondria'     => 7.8,  'ribosom'      => 8.0,
        'mitosis'         => 7.5,  'meiosis'      => 7.5,
        'bakteri'         => 6.0,  'virus'        => 5.5,
        'fungi'           => 6.5,  'ekolog'       => 6.5,
        'fotorespiras'    => 8.0,  'kloroplas'    => 8.0,
        'stomata'         => 7.8,  'xantofil'     => 9.0,

        // ── KIMIA / FISIKA (4.5–8.5) ─────────────────────────────
        'elektron'        => 5.8,  'atom'         => 5.0,
        'molekul'         => 5.5,  'ion'          => 4.8,
        'reaks'           => 4.5,  // reaksi
        'senyawa'         => 5.5,  'larut'        => 4.0, // larutan
        'asam'            => 4.2,  'basa'         => 4.2,
        'kalor'           => 5.8,  'gelombang'    => 5.2,
        'frekuens'        => 5.8,  // frekuensi
        'amplitudo'       => 7.2,  'tekan'        => 3.8, // tekanan
        'massa'           => 4.5,  'volume'       => 4.5,
        'suhu'            => 4.2,  'ikat'         => 4.5, // ikatan
        'oksidas'         => 6.5,  'reduksi'      => 6.5,
        'elektrolit'      => 6.8,  'isomer'       => 7.5,
        'polimer'         => 7.0,  'katalis'      => 6.5,
        'titrasi'         => 7.5,  'molar'        => 6.5,
        'stoikiometri'    => 8.0,  'entalpi'      => 7.0,
        'entropi'         => 7.0,  'termokimia'   => 8.0,
        'percepatan'      => 4.5,  'gaya'         => 4.2,
        'momentum'        => 5.5,  'impuls'       => 6.0,
        'vektor'          => 5.2,  'medan'        => 5.5,
        'potensial'       => 5.0,  'kapasitor'    => 7.0,
        'resistor'        => 7.2,  'transistor'   => 7.5,
        'semikonduktor'   => 7.8,  'spektrum'     => 6.0,
        'difraksi'        => 7.5,  'polarisas'    => 7.0, // polarisasi
        'interferens'     => 7.5,  'fus'          => 5.5, // difusi (bad stem tapi konsisten)

        // ── IPS / SOSIAL (4.0–7.5) ───────────────────────────────
        'demokrasi'       => 5.5,  'konstitusi'   => 6.0,
        'legislatif'      => 6.5,  'eksekutif'    => 6.5,
        'yudikatif'       => 6.8,  'kedaulatan'   => 6.2,
        'pancasila'       => 7.0,  'ekonomi'      => 4.5,
        'politik'         => 5.0,  'hukum'        => 4.8,
        'pemerintah'      => 5.0,  'negara'       => 4.2,
        'bangsa'          => 4.5,  'budaya'       => 4.0,
        'masyarakat'      => 4.0,  'inflasi'      => 6.5,
        'deflasi'         => 7.0,  'resesi'       => 6.5,
        'fiskal'          => 6.8,  'moneter'      => 6.5,
        'investasi'       => 5.5,  'ekspor'       => 5.5,
        'impor'           => 5.5,  'surplus'      => 6.0,
        'defisit'         => 6.0,  'sosiologi'    => 6.5,

        // ── MATEMATIKA / CS (4.0–8.0) ────────────────────────────
        'integral'        => 6.0,  'diferensial'  => 6.5,
        'matriks'         => 5.5,  'probabilitas' => 6.0,
        'statistik'       => 4.5,  'regresi'      => 5.5,
        'fungs'           => 4.5,  // fungsi (stripped i)
        'logaritma'       => 6.5,  'rekursi'      => 7.0,
        'iterasi'         => 5.8,  'kompleksitas' => 6.2,
        'optimasi'        => 5.5,  'kompilasi'    => 7.0,
        'sorting'         => 6.5,  'searching'    => 6.0,
        'hash'            => 6.0,  'trigonometri' => 7.0,

        // ── PENDIDIKAN / KOGNITIF (3.5–6.5) ─────────────────────
        'kurikulum'       => 5.5,  'silabus'      => 6.0,
        'pedagogi'        => 6.5,  'andragogi'    => 8.0,
        'kognitif'        => 6.0,  'afektif'      => 6.2,
        'psikomotor'      => 6.5,  'taksonomi'    => 7.0,
        'konstruktiv'     => 6.5,  'behavioris'   => 6.5,

        // ── STEMS RUSAK TAPI KONSISTEN ───────────────────────────
        'mens'            => 3.5,  // dimensi → mens
        'tabolisme'       => 6.5,  // metabolisme (ulang untuk safety)
        'per'             => 1.8,  // peran → per (stem pendek, bobot rendah)
    ];
}

// ================================================================
// BLENDED IDF: max(corpus_idf, pretrained_idf)
// ================================================================

function compute_idf_blended(array $docs, array $vocab): array {
    $idf_corpus    = compute_idf($docs, $vocab);
    $idf_pretrained = get_pretrained_idf();
    $vocab_flip    = array_flip($vocab);

    foreach ($idf_corpus as $i => &$val) {
        $word = $vocab_flip[$i] ?? '';
        if (isset($idf_pretrained[$word])) {
            $val = max($val, $idf_pretrained[$word]);
        }
    }
    unset($val);
    return $idf_corpus;
}

// ================================================================
// JACCARD SIMILARITY
// ================================================================

function jaccard_similarity(string $doc1, string $doc2): float {
    $w1 = array_unique(array_filter(explode(' ', $doc1)));
    $w2 = array_unique(array_filter(explode(' ', $doc2)));
    if (empty($w1) && empty($w2)) return 1.0;
    $intersection = count(array_intersect($w1, $w2));
    $union        = count(array_unique(array_merge($w1, $w2)));
    return $union > 0 ? round($intersection / $union, 6) : 0.0;
}

// ================================================================
// INDONESIAN STEMMER (Nazief-Adriani Simplified)
// ================================================================

function stem_indonesian(string $word): string {
    static $cache = [];
    if (isset($cache[$word])) return $cache[$word];

    $w = $word;

    foreach (['kan', 'lah', 'kah', 'pun', 'nya', 'an', 'mu', 'ku', 'i'] as $suf) {
        if (str_ends_with($w, $suf) && strlen($w) - strlen($suf) >= 3) {
            $w = substr($w, 0, -strlen($suf));
            break;
        }
    }

    $prefixes = [
        ['meng', 4, false],
        ['meny', 4, true],
        ['mem',  3, false],
        ['men',  3, false],
        ['me',   2, false],
        ['peng', 4, false],
        ['peny', 4, true],
        ['pem',  3, false],
        ['pen',  3, false],
        ['per',  3, false],
        ['pe',   2, false],
        ['ber',  3, false],
        ['ter',  3, false],
        ['di',   2, false],
        ['ke',   2, false],
        ['se',   2, false],
    ];

    foreach ($prefixes as [$pfx, $len, $restore_s]) {
        if (str_starts_with($w, $pfx)) {
            $candidate = $restore_s ? 's' . substr($w, $len) : substr($w, $len);
            if (is_valid_stem($candidate)) {
                $w = $candidate;
                break;
            }
        }
    }

    return $cache[$word] = $w;
}

function is_valid_stem(string $s): bool {
    if (strlen($s) < 3) return false;
    $vowels = ['a', 'e', 'i', 'o', 'u'];
    if (in_array($s[0], $vowels)) return true;
    if (strlen($s) >= 2 && in_array($s[1], $vowels)) return true;
    return false;
}

function get_stem_examples(string $text, int $n = 5): array {
    $text  = mb_strtolower($text, 'UTF-8');
    $text  = preg_replace('/[^a-z\s]/u', ' ', $text);
    $words = array_unique(array_filter(preg_split('/\s+/', $text), fn($w) => strlen($w) > 3));
    $out   = [];
    foreach ($words as $w) {
        $s = stem_indonesian($w);
        if ($s !== $w) {
            $out[$w] = $s;
            if (count($out) >= $n) break;
        }
    }
    return $out;
}

// ================================================================
// SOFT COMPLETENESS
// ================================================================

function soft_completeness(float $sim, float $threshold, float $k = 15.0): float {
    $midpoint = $threshold - 0.1;
    return 1.0 / (1.0 + exp(-$k * ($sim - $midpoint)));
}

// ================================================================
// KEYWORD COVERAGE
// ================================================================

function keyword_coverage(string $jawaban_bersih, string $poin_bersih, array $vocab, array $idf, int $top_n = 5): array {
    $tokens_poin = count(array_filter(explode(' ', $poin_bersih)));
    $top_n = min($top_n, max(1, $tokens_poin));
    if ($tokens_poin === 0) return ['ratio' => 0.0, 'matched' => [], 'missing' => [], 'total' => 0];

    $kunci_tokens = get_top_tfidf($poin_bersih, $vocab, $idf, $top_n);
    $jawaban_set  = array_flip(array_filter(explode(' ', $jawaban_bersih)));
    $matched = $missing = [];
    foreach (array_keys($kunci_tokens) as $tok) {
        if (isset($jawaban_set[$tok])) $matched[] = $tok;
        else $missing[] = $tok;
    }
    $total = count($kunci_tokens);
    return [
        'ratio'   => $total > 0 ? round(count($matched) / $total, 4) : 0.0,
        'matched' => $matched,
        'missing' => $missing,
        'total'   => $total,
    ];
}

// ================================================================
// BM25
// ================================================================

function compute_bm25(string $query, string $doc, array $vocab, array $idf, float $k1 = 1.5, float $b = 0.75, float $avg_len = 1.0): float {
    $doc_words = array_filter(explode(' ', $doc));
    $doc_len   = count($doc_words);
    if ($doc_len === 0) return 0.0;

    $tf_raw = [];
    foreach ($doc_words as $w) {
        if (isset($vocab[$w])) {
            $idx = $vocab[$w];
            $tf_raw[$idx] = ($tf_raw[$idx] ?? 0) + 1;
        }
    }

    $query_words = array_unique(array_filter(explode(' ', $query)));
    $score = 0.0;
    foreach ($query_words as $w) {
        if (!isset($vocab[$w])) continue;
        $idx = $vocab[$w];
        $tf  = $tf_raw[$idx] ?? 0;
        if ($tf === 0) continue;
        $idf_val  = $idf[$idx] ?? 1.0;
        $tf_norm  = ($tf * ($k1 + 1)) / ($tf + $k1 * (1 - $b + $b * $doc_len / max(1.0, $avg_len)));
        $score   += $idf_val * $tf_norm;
    }
    return $score;
}

// ================================================================
// TF-IDF
// ================================================================

function build_vocab(array $docs): array {
    $vocab = [];
    foreach ($docs as $doc) {
        foreach (explode(' ', $doc) as $w) {
            if ($w !== '') $vocab[$w] = true;
        }
    }
    $vocab = array_keys($vocab);
    sort($vocab);
    return array_flip($vocab);
}

function compute_tf(string $doc, array $vocab): array {
    $words = array_filter(explode(' ', $doc));
    $total = count($words);
    $tf    = array_fill(0, count($vocab), 0.0);
    if ($total === 0) return $tf;
    foreach ($words as $w) {
        if (isset($vocab[$w])) $tf[$vocab[$w]] += 1.0;
    }
    return array_map(fn($v) => $v / $total, $tf);
}

function compute_idf(array $docs, array $vocab): array {
    $N  = count($docs);
    $df = array_fill(0, count($vocab), 0);
    foreach ($docs as $doc) {
        $unik = array_unique(array_filter(explode(' ', $doc)));
        foreach ($unik as $w) {
            if (isset($vocab[$w])) $df[$vocab[$w]]++;
        }
    }
    $idf = [];
    foreach ($df as $i => $d) {
        $idf[$i] = $d > 0 ? log(($N + 1) / ($d + 1)) + 1 : 1.0;
    }
    return $idf;
}

function compute_tfidf(string $doc, array $vocab, array $idf): array {
    $tf    = compute_tf($doc, $vocab);
    $tfidf = [];
    foreach ($tf as $i => $v) $tfidf[$i] = $v * $idf[$i];
    return $tfidf;
}

function get_top_tfidf(string $doc, array $vocab, array $idf, int $n = 10): array {
    $tfidf      = compute_tfidf($doc, $vocab, $idf);
    $vocab_flip = array_flip($vocab);
    $pairs      = [];
    foreach ($tfidf as $i => $v) {
        if ($v > 0) $pairs[$vocab_flip[$i]] = round($v, 4);
    }
    arsort($pairs);
    return array_slice($pairs, 0, $n, true);
}

// ================================================================
// COSINE SIMILARITY
// ================================================================

function cosine_similarity(array $a, array $b): float {
    $dot    = dot_product($a, $b);
    $norm_a = vektor_norm($a);
    $norm_b = vektor_norm($b);
    if ($norm_a == 0.0 || $norm_b == 0.0) return 0.0;
    return max(-1.0, min(1.0, $dot / ($norm_a * $norm_b)));
}

function dot_product(array $a, array $b): float {
    $sum = 0.0;
    $len = min(count($a), count($b));
    for ($i = 0; $i < $len; $i++) $sum += $a[$i] * $b[$i];
    return $sum;
}

function vektor_norm(array $a): float {
    $sum = 0.0;
    foreach ($a as $v) $sum += $v * $v;
    return sqrt($sum);
}

// ================================================================
// PRE-PROCESSING
// ================================================================

function preprocess_text(string $text): string {
    $text = mb_strtolower($text, 'UTF-8');
    $text = preg_replace('/https?:\/\/\S+/', '', $text);
    $text = preg_replace('/[^a-z\s]/u', ' ', $text);
    $stopwords = [
        'yang','dan','di','ke','dari','ini','itu','dengan','untuk','pada',
        'adalah','atau','jika','karena','namun','sudah','akan','bisa','juga',
        'oleh','sehingga','agar','dalam','tidak','ada','lebih','sangat','dapat',
        'harus','telah','nya','pun','lagi','atas','bawah','antara','setelah',
        'sebelum','saat','ketika','maka','saja','seperti','bahwa','tentang',
        'hal','cara','kita','kami','mereka','dia','ia','anda','saya',
        'kamu','kalian','sebuah','suatu','setiap','semua','seluruh','para',
        'bagi','demi','hingga','sampai','sejak','selama','terhadap',
        'jadi','menjadi','sedang','sering','selalu','masih','hanya','jangan',
        'bukan','tanpa','walau','meski','walaupun','meskipun','kemudian','lalu',
        'terus','bahkan','justru','yaitu','yakni','misalnya','contohnya',
        'lah','kah','si','sang','satu','dua','tiga','empat','lima',
        'enam','tujuh','delapan','sembilan','sepuluh','pertama','kedua','ketiga',
        'dsb','dll','dst','namun','tetapi','tapi','melainkan','serta','maupun',
        'paling','sekali','terlalu','cukup','agak','hampir','baru','memang',
        'malah','apalagi','terlebih','begitu','demikian','tersebut','adapun',
        'apabila','bilamana','apakah','mengapa','bagaimana','dimana','kapan',
        'siapa','mana','berapa','seberapa','sebagai','dengan','secara','melalui',
        'merupakan','terdapat','ditemukan','dilakukan','digunakan','termasuk',
    ];
    $words = preg_split('/\s+/', $text);
    $words = array_filter($words, fn($w) => strlen($w) > 1 && !in_array($w, $stopwords));
    $words = array_map('stem_indonesian', $words);
    $words = array_filter($words, fn($w) => strlen($w) > 1);
    return trim(implode(' ', $words));
}

// ================================================================
// LABEL
// ================================================================

function get_label(float $sim): string {
    if ($sim >= 0.80) return 'Sangat Mirip';
    if ($sim >= 0.60) return 'Mirip';
    if ($sim >= 0.40) return 'Cukup Mirip';
    if ($sim >= 0.20) return 'Kurang Mirip';
    return 'Tidak Mirip';
}

// ================================================================
// FILE UPLOAD
// ================================================================

function extract_text_from_file(array $file): string {
    if ($file['size'] > 5 * 1024 * 1024) send_error(400, 'File terlalu besar. Maksimal 5 MB.');
    $mime = mime_content_type($file['tmp_name']);
    if ($mime === 'application/pdf' && function_exists('exec')) {
        $tmp = tempnam(sys_get_temp_dir(), 'pdf_');
        exec('pdftotext ' . escapeshellarg($file['tmp_name']) . ' ' . escapeshellarg($tmp));
        if (file_exists($tmp)) { $t = file_get_contents($tmp); unlink($tmp); if (trim($t)) return $t; }
    }
    if (in_array($mime, ['image/jpeg','image/png','image/webp']) && function_exists('exec')) {
        $tmp = tempnam(sys_get_temp_dir(), 'ocr_');
        exec('tesseract ' . escapeshellarg($file['tmp_name']) . ' ' . escapeshellarg($tmp) . ' -l ind 2>/dev/null');
        $f = $tmp . '.txt';
        if (file_exists($f)) { $t = file_get_contents($f); unlink($f); if (trim($t)) return $t; }
    }
    return '';
}

// ================================================================
// UTILITY
// ================================================================

function get_input(): array {
    $raw = file_get_contents('php://input');
    if (!empty($raw)) {
        $d = json_decode($raw, true);
        if (json_last_error() === JSON_ERROR_NONE) return $d;
    }
    return $_POST;
}

function send_error(int $code, string $msg): void {
    http_response_code($code);
    echo json_encode(['status' => 'error', 'code' => $code, 'message' => $msg], JSON_UNESCAPED_UNICODE);
    exit;
}
