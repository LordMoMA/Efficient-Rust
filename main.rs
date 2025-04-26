// Example 1: Zero-Cost Abstractions with Iterators vs Manual Loops
// In many languages, using higher-order functions comes with a performance penalty
// In Rust, the iterator version often compiles to the same efficient machine code as manual loops

// Manual loop approach (what you might do in other languages for performance)
fn sum_even_squares_manual(numbers: &[i32]) -> i32 {
    let mut sum = 0;
    for i in 0..numbers.len() {
        let num = numbers[i];
        if num % 2 == 0 {
            sum += num * num;
        }
    }
    sum
}

// Iterator approach (more readable, yet just as fast in Rust)
fn sum_even_squares_iterator(numbers: &[i32]) -> i32 {
    numbers
        .iter()
        .filter(|&&x| x % 2 == 0)
        .map(|&x| x * x)
        .sum()
}

// Example 2: Memory Management Without Garbage Collection
// Demonstrating how Rust's ownership system eliminates common performance issues

// In many languages: Hidden allocations, copies, and GC pressure
fn process_text_other_lang(text: String) -> String {
    // Implicitly creates multiple copies and temporary allocations
    let lines = text.split('\n'); // Hidden allocation in many languages
    let filtered = lines.filter(|line| !line.trim().is_empty()); // Another allocation
    let processed = filtered.map(|line| line.trim().to_uppercase()); // More allocations
    processed.collect::<Vec<_>>().join("\n") // Final allocation
}

// In Rust: Explicit and efficient with zero hidden costs
fn process_text_rust(text: &str) -> String {
    text.split('\n')
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.trim().to_uppercase())
        .collect::<Vec<_>>()
        .join("\n")
}

// Example 3: String Handling - Slices vs Owned Data
// Demonstrating how Rust's type system guides you toward efficient choices

// Inefficient approach - unnecessary ownership transfers and allocations
fn extract_domain_inefficient(email: String) -> String {
    let parts: Vec<&str> = email.split('@').collect();
    if parts.len() != 2 {
        return String::new();
    }
    parts[1].to_string()
}

// Efficient approach - using references to avoid allocations
fn extract_domain_efficient(email: &str) -> &str {
    if let Some(at_pos) = email.find('@') {
        &email[at_pos + 1..]
    } else {
        ""
    }
}

// Example 4: Compile-time Evaluation
// Rust's const generics and const functions enable computation at compile time

// Computing Fibonacci at runtime (like most languages would)
fn fibonacci_runtime(n: u32) -> u64 {
    if n <= 1 {
        return n as u64;
    }

    let mut a = 0;
    let mut b = 1;
    for _ in 2..=n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    b
}

// Computing Fibonacci at compile time, Without const, It’s Always Runtime
const fn fibonacci_compile_time(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        n => {
            let mut a = 0;
            let mut b = 1;
            let mut i = 2;
            while i <= n {
                let temp = a + b;
                a = b;
                b = temp;
                i += 1;
            }
            b
        }
    }
}

// Using compile-time evaluation
const FIB_10: u64 = fibonacci_compile_time(10); // Computed at compile time!

static FIB_10: u64 = fibonacci_compile_time(10); // Compile-time

let arr: [i32; fibonacci_compile_time(5) as usize] = [0; 5]; // Compile-time

let fib_11 = fibonacci_compile_time(11); // Runtime

// Example 5: Safe Concurrency Without Performance Overhead
// Demonstrating how Rust's ownership rules enable efficient parallelism

use std::thread;

// In many languages: Locks, synchronization, or copying data
fn parallel_sum_other_langs(data: &[i32]) -> i32 {
    // Would typically involve locks, mutexes, or copying data
    // Often leads to contention or excessive memory usage
    // Implementation omitted for brevity
    0 // Placeholder
}

// In Rust: Efficient parallelism with compile-time safety
fn parallel_sum_rust(data: &[i32]) -> i32 {
    if data.len() < 1000 {
        // For small arrays, just do it sequentially
        return data.iter().sum();
    }

    // For larger arrays, split the work
    let mid = data.len() / 2;
    let (left, right) = data.split_at(mid);

    // Process each half in parallel with zero-copy slices
    let left_handle = thread::spawn(move || left.iter().sum::<i32>());

    let right_sum = right.iter().sum::<i32>();
    let left_sum = left_handle.join().unwrap();

    left_sum + right_sum
}

// Example 6: Avoiding Dynamic Dispatch with Generics and Traits
// Showing how Rust can eliminate runtime type checking overhead

// In many languages: Runtime polymorphism with performance cost
trait ShapeOtherLang {
    fn area(&self) -> f64;
}

struct CircleOtherLang {
    radius: f64,
}

impl ShapeOtherLang for CircleOtherLang {
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
}

// Would incur virtual dispatch overhead in most languages
fn total_area_dynamic(shapes: &[Box<dyn ShapeOtherLang>]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}

// In Rust: Static dispatch with generics - zero overhead
trait Shape {
    fn area(&self) -> f64;
}

struct Circle {
    radius: f64,
}

impl Shape for Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
}

// No runtime dispatch needed - monomorphization creates optimized code for each type
fn total_area<T: Shape>(shapes: &[T]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}

// Example 7: Move Semantics - Avoiding Hidden Copies
// Demonstrating how Rust's move semantics prevent unexpected performance issues

struct LargeData {
    // Imagine this contains megabytes of data
    payload: Vec<u8>,
}

// In many languages: Implicit, expensive copying
fn process_data_other_lang(data: LargeData) -> LargeData {
    // Many languages would implicitly copy 'data' here
    // This could be very expensive but hidden from the developer
    LargeData {
        payload: data.payload, // More copying!
    }
}

// In Rust: Explicit transfers of ownership
fn process_data_rust(mut data: LargeData) -> LargeData {
    // Ownership is moved, no hidden copies
    // We can modify in place for efficiency
    data.payload.push(42);
    data // Transfer ownership back to caller
}

// Or, even more efficient when appropriate
fn process_data_rust_ref(data: &mut LargeData) {
    // No ownership transfer at all, just borrowed access
    data.payload.push(42);
}

// Example 8: Inlining and Optimizing Small Functions
// Demonstrating how Rust's compiler aggressively optimizes code

// This small function will be inlined automatically
#[inline]
fn add_one(x: i32) -> i32 {
    x + 1
}

fn process_numbers(numbers: &[i32]) -> Vec<i32> {
    // The compiler will inline add_one, eliminating function call overhead
    numbers.iter().map(|&x| add_one(x)).collect()
}

// Example 9: Stack vs Heap Allocation
// Showing how Rust encourages stack allocation for better performance

// In many languages: Hidden heap allocations
fn create_array_other_lang() -> Vec<i32> {
    // Most languages would heap-allocate this array
    vec![1, 2, 3, 4, 5]
}

// In Rust: Explicit choice between stack and heap
fn create_array_stack() -> [i32; 5] {
    // Stack allocated - no allocation overhead, better cache locality
    [1, 2, 3, 4, 5]
}

fn create_array_heap() -> Vec<i32> {
    // Explicitly heap allocated when needed
    vec![1, 2, 3, 4, 5]
}

// Example 10: Compile-time Bounds Checking Elimination
// Showing how Rust's compiler can eliminate unnecessary bounds checks

fn sum_array(arr: &[i32]) -> i32 {
    let mut sum = 0;

    // The compiler can prove this loop is safe and eliminate bounds checks
    for i in 0..arr.len() {
        sum += arr[i]; // No runtime bounds check needed!
    }

    sum
}

// More advanced: Using unsafe when appropriate with clear safety boundaries
fn sum_array_optimized(arr: &[i32]) -> i32 {
    let mut sum = 0;

    // Explicitly avoiding bounds checks when we're certain it's safe
    unsafe {
        for i in 0..arr.len() {
            sum += *arr.get_unchecked(i);
        }
    }

    sum
}

// Example 11: Efficient Error Handling with Result
// Showing how Rust's Result type provides zero-cost error handling

// In many languages: Exceptions with significant overhead
// try {
//     doSomething();
//     doSomethingElse();
// } catch (Exception e) {
//     handleError(e);
// }

// In Rust: Zero-cost abstraction for error handling
fn process_data_result(input: &str) -> Result<i32, &'static str> {
    // The ? operator provides elegant error handling with no performance penalty
    let value = parse_value(input)?;
    let processed = transform_value(value)?;
    Ok(processed)
}

fn parse_value(input: &str) -> Result<i32, &'static str> {
    match input.parse::<i32>() {
        Ok(n) => Ok(n),
        Err(_) => Err("Failed to parse input"),
    }
}

fn transform_value(n: i32) -> Result<i32, &'static str> {
    if n < 0 {
        Err("Value cannot be negative")
    } else {
        Ok(n * 2)
    }
}

// Example 12: Efficient Enums with Pattern Matching
// Demonstrating how Rust's enums avoid the overhead of OOP hierarchies

// In OOP languages: Virtual dispatch and downcasting
// class Shape { public: virtual double area() = 0; };
// class Circle : public Shape { ... };
// class Rectangle : public Shape { ... };
//
// double process(Shape* shape) {
//     if (Circle* c = dynamic_cast<Circle*>(shape)) {
//         // Circle-specific logic
//     } else if (Rectangle* r = dynamic_cast<Rectangle*>(shape)) {
//         // Rectangle-specific logic
//     }
// }

// In Rust: Efficient tagged unions with pattern matching
enum Shape {
    Circle { radius: f64 },
    Rectangle { width: f64, height: f64 },
    Triangle { base: f64, height: f64 },
}

impl Shape {
    fn area(&self) -> f64 {
        match self {
            // No runtime type checking or downcasting needed
            Shape::Circle { radius } => std::f64::consts::PI * radius * radius,
            Shape::Rectangle { width, height } => width * height,
            Shape::Triangle { base, height } => 0.5 * base * height,
        }
    }
}

// Example 13: SIMD Vectorization
// Showing how Rust allows explicit SIMD instructions for maximum performance

use std::arch::x86_64::*;

// Safe wrapper around SIMD operations
#[cfg(target_arch = "x86_64")]
fn sum_vector_simd(data: &[f32]) -> f32 {
    let len = data.len();
    let remainder = len % 4;
    let simd_len = len - remainder;

    let mut sum = 0.0;

    // Process 4 elements at a time using SIMD
    if is_x86_feature_detected!("sse2") {
        unsafe {
            let mut sum_vec = _mm_setzero_ps();

            for i in (0..simd_len).step_by(4) {
                let v = _mm_loadu_ps(data[i..].as_ptr());
                sum_vec = _mm_add_ps(sum_vec, v);
            }

            // Extract and sum the 4 float values from the vector
            let mut temp = [0.0f32; 4];
            _mm_storeu_ps(temp.as_mut_ptr(), sum_vec);
            sum = temp.iter().sum::<f32>();
        }
    }

    // Add the remaining elements
    for i in simd_len..len {
        sum += data[i];
    }

    sum
}

// Example 14: Efficient Binary Serialization with Zero-Copy
// Demonstrating how Rust can parse binary data without unnecessary copies

#[repr(C, packed)]
struct NetworkPacket {
    packet_type: u8,
    flags: u8,
    length: u16,
    data: [u8; 1024],
}

// In many languages: Parse by copying bytes
// function parsePacket(buffer) {
//     const type = buffer.readUInt8(0);
//     const flags = buffer.readUInt8(1);
//     const length = buffer.readUInt16LE(2);
//     const data = Buffer.alloc(length);
//     buffer.copy(data, 0, 4, 4 + length);
//     return { type, flags, length, data };
// }

// In Rust: Zero-copy parsing with memory mapping
fn parse_packet(buffer: &[u8]) -> Option<&NetworkPacket> {
    if buffer.len() < std::mem::size_of::<NetworkPacket>() {
        return None;
    }

    // Safely interpret the buffer as a NetworkPacket without copying
    unsafe {
        let packet = &*(buffer.as_ptr() as *const NetworkPacket);
        Some(packet)
    }
}

// Example 15: Memory Pooling for Frequent Allocations
// Showing explicit memory reuse patterns that other languages handle via GC

struct ObjectPool<T> {
    objects: Vec<Option<T>>,
}

impl<T> ObjectPool<T> {
    fn new(capacity: usize) -> Self {
        let mut objects = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            objects.push(None);
        }
        ObjectPool { objects }
    }

    fn acquire(&mut self, create_fn: impl FnOnce() -> T) -> usize {
        // Find an empty slot or create new
        for (idx, slot) in self.objects.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(create_fn());
                return idx;
            }
        }

        // No empty slots, add a new one
        let idx = self.objects.len();
        self.objects.push(Some(create_fn()));
        idx
    }

    fn release(&mut self, idx: usize) {
        if idx < self.objects.len() {
            self.objects[idx] = None;
        }
    }

    fn get(&self, idx: usize) -> Option<&T> {
        if idx >= self.objects.len() {
            return None;
        }
        self.objects[idx].as_ref()
    }
}

// Example 16: Efficient Binary Search with No Dynamic Allocation
// Demonstrating allocation-free algorithms

// Many languages: Hidden allocations in standard library
// function binarySearch(arr, target) {
//     let left = 0;
//     let right = arr.length - 1;
//     while (left <= right) {
//         const mid = Math.floor((left + right) / 2);
//         if (arr[mid] === target) return mid;
//         if (arr[mid] < target) left = mid + 1;
//         else right = mid - 1;
//     }
//     return -1;
// }

// In Rust: Zero allocation binary search
fn binary_search<T: Ord>(slice: &[T], target: &T) -> Option<usize> {
    // No allocation, works on any ordered slice
    slice.binary_search(target).ok()
}

// Custom implementation that makes zero allocations
fn binary_search_manual<T: Ord>(slice: &[T], target: &T) -> Option<usize> {
    let mut left = 0;
    let mut right = slice.len();

    while left < right {
        let mid = left + (right - left) / 2;
        match slice[mid].cmp(target) {
            std::cmp::Ordering::Equal => return Some(mid),
            std::cmp::Ordering::Less => left = mid + 1,
            std::cmp::Ordering::Greater => right = mid,
        }
    }

    None
}

// Example 17: Cache-Friendly Data Structures
// Demonstrating how Rust enables designing for the memory hierarchy

// Structure of Arrays (SoA) vs Array of Structures (AoS)
// SoA is often more cache-friendly for batch processing

// Array of Structures (traditional OOP approach)
struct ParticleAoS {
    position_x: f32,
    position_y: f32,
    position_z: f32,
    velocity_x: f32,
    velocity_y: f32,
    velocity_z: f32,
}

// Structure of Arrays (cache-optimized approach)
struct ParticleSystemSoA {
    position_x: Vec<f32>,
    position_y: Vec<f32>,
    position_z: Vec<f32>,
    velocity_x: Vec<f32>,
    velocity_y: Vec<f32>,
    velocity_z: Vec<f32>,
}

impl ParticleSystemSoA {
    fn update_positions(&mut self, dt: f32) {
        // Process each component in a tight loop - better cache utilization
        for i in 0..self.position_x.len() {
            self.position_x[i] += self.velocity_x[i] * dt;
        }

        for i in 0..self.position_y.len() {
            self.position_y[i] += self.velocity_y[i] * dt;
        }

        for i in 0..self.position_z.len() {
            self.position_z[i] += self.velocity_z[i] * dt;
        }
    }
}

// Example 18: Custom Allocators
// Showing how Rust allows complete control over memory allocation strategies

use std::alloc::{GlobalAlloc, Layout, System};

// A custom allocator that tracks allocations
struct TracingAllocator;

unsafe impl GlobalAlloc for TracingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        // In a real implementation, we'd log this allocation
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        // In a real implementation, we'd log this deallocation
    }
}

// To use: #[global_allocator] static ALLOCATOR: TracingAllocator = TracingAllocator;

// Example 19: Efficient Text Processing with No Allocations
// Demonstrating zero-allocation string manipulation

fn count_words(text: &str) -> usize {
    // No allocations needed - we just iterate through the string once
    text.split_whitespace().count()
}

fn is_palindrome(text: &str) -> bool {
    // Convert to lowercase without allocation by working on UTF-8 bytes
    let bytes: Vec<_> = text
        .bytes()
        .filter(|b| b.is_ascii_alphabetic())
        .map(|b| b.to_ascii_lowercase())
        .collect();

    let len = bytes.len();
    for i in 0..len / 2 {
        if bytes[i] != bytes[len - 1 - i] {
            return false;
        }
    }

    true
}

// Example 20: Memory-Mapped Files for Large Data Processing
// Showing efficient handling of data larger than RAM

use memmap2::MmapOptions;
use std::fs::File;

fn count_lines_in_large_file(path: &str) -> std::io::Result<usize> {
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    // Process gigabytes of data without loading it all into memory
    let mut count = 0;
    for byte in &mmap {
        if *byte == b'\n' {
            count += 1;
        }
    }

    Ok(count)
}

// Example 21: Efficient Graph Traversal with Arena Allocation
// Demonstrating how Rust enables custom allocation strategies for complex data structures

use std::cell::Cell;

// An arena-based graph representation that avoids individual node allocations
struct NodeArena {
    nodes: Vec<Node>,
}

struct Node {
    value: i32,
    // Using indices instead of pointers/references
    edges: Vec<usize>,
    visited: Cell<bool>,
}

impl NodeArena {
    fn new() -> Self {
        NodeArena { nodes: Vec::new() }
    }
    
    fn add_node(&mut self, value: i32) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            value,
            edges: Vec::new(),
            visited: Cell::new(false),
        });
        idx
    }
    
    fn add_edge(&mut self, from: usize, to: usize) {
        if from < self.nodes.len() && to < self.nodes.len() {
            self.nodes[from].edges.push(to);
        }
    }
    
    // Depth-first search with no recursion and minimal allocations
    fn dfs(&self, start: usize) -> Vec<i32> {
        let mut result = Vec::new();
        let mut stack = Vec::new();
        
        // Reset visited flags
        for node in &self.nodes {
            node.visited.set(false);
        }
        
        stack.push(start);
        while let Some(idx) = stack.pop() {
            if idx >= self.nodes.len() || self.nodes[idx].visited.get() {
                continue;
            }
            
            let node = &self.nodes[idx];
            node.visited.set(true);
            result.push(node.value);
            
            // Push neighbors in reverse order for correct traversal
            for &edge in node.edges.iter().rev() {
                stack.push(edge);
            }
        }
        
        result
    }
}

// Example 22: Non-Allocating Iterator Adapters
// Showing how Rust enables complex transformations without allocations

fn parse_logs_efficient<'a>(logs: &'a str) -> impl Iterator<Item = &'a str> + 'a {
    // All these transformations happen lazily with no intermediate allocations
    logs.lines()
        .filter(|line| line.starts_with("ERROR:"))
        .map(|line| line.trim_start_matches("ERROR:").trim())
        // Only when collected will any allocation happen
}

// Contrast with inefficient approach in many languages:
fn parse_logs_inefficient(logs: &str) -> Vec<String> {
    let mut result = Vec::new();
    
    for line in logs.lines() {
        if line.starts_with("ERROR:") {
            let processed = line.trim_start_matches("ERROR:").trim().to_string();
            result.push(processed);
        }
    }
    
    result
}

// Example 23: Type-Level State Machines
// Using the type system to enforce correctness at compile time

// States for a TCP connection
struct Closed;
struct Listening;
struct Connected;

// Type-safe TCP socket that enforces the correct state transitions
struct TcpSocket<State> {
    socket_fd: i32,
    _state: std::marker::PhantomData<State>,
}

impl TcpSocket<Closed> {
    fn new() -> Self {
        // Create socket
        TcpSocket {
            socket_fd: 0, // Simplified for example
            _state: std::marker::PhantomData,
        }
    }
    
    fn listen(self, port: u16) -> TcpSocket<Listening> {
        // Bind and listen
        println!("Listening on port {}", port);
        
        TcpSocket {
            socket_fd: self.socket_fd,
            _state: std::marker::PhantomData,
        }
    }
}

impl TcpSocket<Listening> {
    fn accept(&self) -> TcpSocket<Connected> {
        // Accept connection
        println!("Connection accepted");
        
        TcpSocket {
            socket_fd: self.socket_fd + 1, // New connection
            _state: std::marker::PhantomData,
        }
    }
}

impl TcpSocket<Connected> {
    fn send(&self, data: &[u8]) -> usize {
        // Send data
        println!("Sending {} bytes", data.len());
        data.len()
    }
    
    fn close(self) -> TcpSocket<Closed> {
        // Close connection
        println!("Connection closed");
        
        TcpSocket {
            socket_fd: self.socket_fd,
            _state: std::marker::PhantomData,
        }
    }
}

// Using the state machine:
fn use_socket() {
    let socket = TcpSocket::new();
    let socket = socket.listen(8080);
    let connected = socket.accept();
    connected.send(b"Hello");
    // This would not compile: socket.accept(); - socket is already moved
    // This would not compile: connected.listen(8080); - wrong state
    let closed = connected.close();
    // This would not compile: connected.send(b"World"); - socket is moved to closed state
}

// Example 24: Compile-Time Dimensional Analysis
// Using const generics for physical units

#[derive(Debug, Clone, Copy)]
struct Quantity<const METER: i8, const SECOND: i8, const KILOGRAM: i8> {
    value: f64,
}

impl<const M: i8, const S: i8, const KG: i8> Quantity<M, S, KG> {
    fn new(value: f64) -> Self {
        Quantity { value }
    }
}

// Addition only works for same units
impl<const M: i8, const S: i8, const KG: i8> std::ops::Add for Quantity<M, S, KG> {
    type Output = Quantity<M, S, KG>;
    
    fn add(self, rhs: Self) -> Self::Output {
        Quantity { value: self.value + rhs.value }
    }
}

// Multiplication changes units according to physics
impl<const M1: i8, const S1: i8, const KG1: i8, const M2: i8, const S2: i8, const KG2: i8> 
std::ops::Mul<Quantity<M2, S2, KG2>> for Quantity<M1, S1, KG1> {
    type Output = Quantity<{M1 + M2}, {S1 + S2}, {KG1 + KG2}>;
    
    fn mul(self, rhs: Quantity<M2, S2, KG2>) -> Self::Output {
        Quantity { value: self.value * rhs.value }
    }
}

// Type aliases for common units
type Meter<T = f64> = Quantity<1, 0, 0>;
type Second<T = f64> = Quantity<0, 1, 0>;
type Kilogram<T = f64> = Quantity<0, 0, 1>;
type MeterPerSecond<T = f64> = Quantity<1, -1, 0>;
type Newton<T = f64> = Quantity<1, -2, 1>; // kg·m/s²

fn physics_calculation() {
    let distance = Meter::new(10.0);
    let time = Second::new(2.0);
    let speed: MeterPerSecond = Quantity::<1, 0, 0>::new(10.0) * Quantity::<0, -1, 0>::new(1.0);
    
    // This would not compile - incompatible units:
    // let error = distance + time;
    
    // But this works fine:
    let distance2 = distance + Meter::new(5.0);
    println!("Total distance: {} meters", distance2.value);
}

// Example 25: Efficient Bitsets with No Heap Allocation
// Fixed-size bitsets that live entirely on the stack

struct StaticBitSet<const N: usize> {
    // Each u64 stores 64 bits
    bits: [u64; (N + 63) / 64],
}

impl<const N: usize> StaticBitSet<N> {
    fn new() -> Self {
        StaticBitSet { bits: [0; (N + 63) / 64] }
    }
    
    fn set(&mut self, idx: usize, value: bool) {
        if idx >= N {
            return;
        }
        
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        
        if value {
            self.bits[word_idx] |= 1u64 << bit_idx;
        } else {
            self.bits[word_idx] &= !(1u64 << bit_idx);
        }
    }
    
    fn get(&self, idx: usize) -> bool {
        if idx >= N {
            return false;
        }
        
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        
        (self.bits[word_idx] & (1u64 << bit_idx)) != 0
    }
    
    fn count_ones(&self) -> usize {
        self.bits.iter().map(|word| word.count_ones() as usize).sum()
    }
}

// Example 26: Lock-Free Concurrent Data Structures
// Using atomic operations for thread-safe data without locks

use std::sync::atomic::{AtomicUsize, Ordering};

struct LockFreeCounter {
    value: AtomicUsize,
}

impl LockFreeCounter {
    fn new() -> Self {
        LockFreeCounter { value: AtomicUsize::new(0) }
    }
    
    fn increment(&self) -> usize {
        // Atomically increment and get previous value
        self.value.fetch_add(1, Ordering::SeqCst)
    }
    
    fn get(&self) -> usize {
        self.value.load(Ordering::SeqCst)
    }
}

// Example 27: Compile-Time Regular Expressions
// Demonstrating constant evaluation of regex patterns

use regex::Regex;

// In many languages: Regex is compiled at runtime
// const REGEX_PATTERN = /^\d{4}-\d{2}-\d{2}$/;
// function isValidDate(date) {
//     return REGEX_PATTERN.test(date);
// }

// In Rust: Regex is compiled at compile time with lazy_static
fn is_valid_date(date: &str) -> bool {
    // This regex is compiled once at program startup
    lazy_static::lazy_static! {
        static ref DATE_REGEX: Regex = Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();
    }
    
    DATE_REGEX.is_match(date)
}

// Even better with regex crate's macro for compile-time regex
fn is_valid_date_optimized(date: &str) -> bool {
    // This regex is compiled at compile time
    static DATE_REGEX: regex::Regex = regex::Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();
    
    DATE_REGEX.is_match(date)
}

// Example 28: Inlined Function Dispatch via Trait Objects
// Showing how traits can be monomorphized for performance

trait Drawable {
    fn draw(&self);
}

struct Circle {
    radius: f64,
}

impl Drawable for Circle {
    fn draw(&self) {
        println!("Drawing circle with radius {}", self.radius);
    }
}

struct Rectangle {
    width: f64,
    height: f64,
}

impl Drawable for Rectangle {
    fn draw(&self) {
        println!("Drawing rectangle {}x{}", self.width, self.height);
    }
}

// Dynamic dispatch (slightly slower)
fn draw_all(shapes: &[&dyn Drawable]) {
    for shape in shapes {
        shape.draw();
    }
}

// Static dispatch (faster, specialized for each type)
fn draw_two<T: Drawable, U: Drawable>(shape1: &T, shape2: &U) {
    shape1.draw();
    shape2.draw();
}

// Example 29: Non-Allocating String Parsing
// Parsing structured data without intermediate allocations

struct LogEntry<'a> {
    timestamp: &'a str,
    level: &'a str,
    message: &'a str,
}

fn parse_log_entry(line: &str) -> Option<LogEntry> {
    // No allocations, just references into the original string
    let mut parts = line.splitn(3, ' ');
    
    let timestamp = parts.next()?;
    let level = parts.next()?;
    let message = parts.next()?;
    
    Some(LogEntry {
        timestamp,
        level,
        message,
    })
}

// Example 30: Efficient Path Searching with A* Algorithm
// Showing idiomatic implementation of complex algorithms

use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

#[derive(Copy, Clone, Eq, PartialEq)]
struct State {
    cost: usize,
    position: (usize, usize),
}

// Custom ordering for priority queue
impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Notice this is reversed for min-heap
        other.cost.cmp(&self.cost)
            .then_with(|| self.position.cmp(&other.position))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn astar(
    grid: &[Vec<bool>],
    start: (usize, usize),
    goal: (usize, usize),
) -> Option<Vec<(usize, usize)>> {
    let rows = grid.len();
    let cols = grid[0].len();
    
    // Priority queue for frontier
    let mut frontier = BinaryHeap::new();
    frontier.push(State { cost: 0, position: start });
    
    // Track visited cells and their costs
    let mut came_from: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    let mut cost_so_far: HashMap<(usize, usize), usize> = HashMap::new();
    cost_so_far.insert(start, 0);
    
    while let Some(State { cost, position }) = frontier.pop() {
        if position == goal {
            // Reconstruct path
            let mut path = vec![goal];
            let mut current = goal;
            while current != start {
                current = came_from[&current];
                path.push(current);
            }
            path.reverse();
            return Some(path);
        }
        
        // Skip if we've found a better path
        if let Some(&existing_cost) = cost_so_far.get(&position) {
            if cost > existing_cost {
                continue;
            }
        }
        
        // Check all neighbors
        let (x, y) = position;
        let neighbors = [
            (x.wrapping_sub(1), y),
            (x + 1, y),
            (x, y.wrapping_sub(1)),
            (x, y + 1),
        ];
        
        for &next in &neighbors {
            if next.0 >= rows || next.1 >= cols || grid[next.0][next.1] {
                // Out of bounds or obstacle
                continue;
            }
            
            let new_cost = cost_so_far[&position] + 1;
            if !cost_so_far.contains_key(&next) || new_cost < cost_so_far[&next] {
                cost_so_far.insert(next, new_cost);
                
                // Priority includes heuristic (Manhattan distance)
                let priority = new_cost + manhattan_distance(next, goal);
                frontier.push(State { cost: priority, position: next });
                came_from.insert(next, position);
            }
        }
    }
    
    None // No path found
}

fn manhattan_distance(a: (usize, usize), b: (usize, usize)) -> usize {
    // Wrapping sub handles unsigned subtraction safely
    let dx = if a.0 > b.0 { a.0 - b.0 } else { b.0 - a.0 };
    let dy = if a.1 > b.1 { a.1 - b.1 } else { b.1 - a.1 };
    dx + dy
}

// Example 31: Branch Prediction Optimization
// Sorting data before processing to enable CPU branch prediction

fn sum_if_positive(data: &[i32]) -> i32 {
    let mut sum = 0;
    
    // Naive approach: branching on each element
    for &x in data {
        if x > 0 {
            sum += x;
        }
    }
    
    sum
}

fn sum_if_positive_optimized(data: &[i32]) -> i32 {
    // First sort the data - makes branch prediction more efficient
    let mut sorted = data.to_vec();
    sorted.sort_unstable();
    
    let mut sum = 0;
    // Now CPU can predict branches better since data is ordered
    for &x in &sorted {
        if x > 0 {
            sum += x;
        }
    }
    
    sum
}

// Example 32: Static vs Dynamic Configuration
// Showing how Rust enables compile-time configuration

// In many languages: Configuration loaded at runtime
// class Config {
//     static getFeatureEnabled(name) {
//         return Config.settings[name] === true;
//     }
// }

// In Rust: Features enabled at compile time with cfg
fn process_data(data: &[u32]) -> Vec<u32> {
    let mut result = Vec::with_capacity(data.len());
    
    for &item in data {
        let processed = if cfg!(feature = "double_values") {
            item * 2
        } else {
            item
        };
        
        result.push(processed);
    }
    
    result
}

// Example 33: Efficient Random Access with Skip Lists
// Showing how to implement complex data structures with optimal performance

struct SkipNode {
    value: i32, 
    // Vector of next pointers at different levels
    forward: Vec<Option<Box<SkipNode>>>,
}

struct SkipList {
    head: Box<SkipNode>,
    max_level: usize,
    current_max_level: usize,
}

impl SkipList {
    fn new(max_level: usize) -> Self {
        SkipList {
            head: Box::new(SkipNode {
                value: i32::MIN,
                forward: vec![None; max_level],
            }),
            max_level,
            current_max_level: 0,
        }
    }
    
    // Randomly generates level for node
    fn random_level(&self) -> usize {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut level = 0;
        
        // P = 0.5
        while rng.gen::<bool>() && level < self.max_level - 1 {
            level += 1;
        }
        
        level
    }
    
    fn insert(&mut self, value: i32) {
        let mut update = vec![None; self.max_level];
        let mut current = &mut self.head;
        
        // Find position to insert at each level
        for i in (0..=self.current_max_level).rev() {
            while let Some(ref mut next) = current.forward[i] {
                if next.value < value {
                    current = next;
                } else {
                    break;
                }
            }
            update[i] = Some(current);
        }
        
        // Generate random level for new node
        let level = self.random_level();
        
        // Update maximum level if needed
        if level > self.current_max_level {
            self.current_max_level = level;
        }
        
        // Create new node
        let mut new_node = Box::new(SkipNode {
            value,
            forward: vec![None; level + 1],
        });
        
        // Update forward pointers
        for i in 0..=level {
            if let Some(ref mut node) = update[i] {
                new_node.forward[i] = node.forward[i].take();
                node.forward[i] = Some(new_node.clone());
            }
        }
    }
    
    fn search(&self, value: i32) -> bool {
        let mut current = &self.head;
        
        // Start from highest level for efficient search
        for i in (0..=self.current_max_level).rev() {
            while let Some(ref next) = current.forward[i] {
                if next.value < value {
                    current = next;
                } else if next.value == value {
                    return true;
                } else {
                    break;
                }
            }
        }
        
        false
    }
}

// Example 34: Specialized Hash Functions
// Customizing hash algorithms for specific data types

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// Domain-specific hashing for IP addresses
struct IpAddress {
    octets: [u8; 4],
}

impl Hash for IpAddress {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // For IP addresses, we can just hash the integer representation
        let ip_as_u32 = u32::from_be_bytes(self.octets);
        ip_as_u32.hash(state);
    }
}

// Example 35: Efficient String Interner
// Deduplicating strings for memory efficiency

struct StringInterner {
    strings: Vec<String>,
    map: std::collections::HashMap<String, usize>,
}

impl StringInterner {
    fn new() -> Self {
        StringInterner {
            strings: Vec::new(),
            map: std::collections::HashMap::new(),
        }
    }
    
    // Intern a string, returning a unique identifier
    fn intern(&mut self, s: &str) -> usize {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        
        let id = self.strings.len();
        self.strings.push(s.to_string());
        self.map.insert(s.to_string(), id);
        id
    }
    
    // Get string by ID
    fn lookup(&self, id: usize) -> Option<&str> {
        self.strings.get(id).map(|s| s.as_str())
    }
}

// Example 36: Constant-Time Operations with Lookup Tables
// Pre-computed results for expensive operations

// Population count (number of set bits) using a lookup table
fn popcount_table(x: u8) -> u8 {
    // Pre-computed lookup table
    const POPCOUNT_TABLE: [u8; 256] = [
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        // ... (full table omitted for brevity)
        4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
    ];
    
    POPCOUNT_TABLE[x as usize]
}

// Example 37: Locality-Sensitive Hashing
// Efficient similarity search with custom hashing

struct MinHasher {
    hash_functions: Vec<fn(u32) -> u32>,
    min_hashes: Vec<u32>,
}

impl MinHasher {
    fn new(num_hashes: usize) -> Self {
        // Create hash functions with different seeds
        let hash_functions: Vec<fn(u32) -> u32> = (0..num_hashes)
            .map(|seed| {
                move |x| {
                    let mut h = x.wrapping_add(seed as u32);
                    h = h.wrapping_mul(0x9e3779b1);
                    h ^= h >> 16;
                    h
                }
            })
            .collect();
        
        MinHasher {
            hash_functions,
            min_hashes: vec![u32::MAX; num_hashes],
        }
    }
    
    fn add(&mut self, item: u32) {
        for (i, hash_fn) in self.hash_functions.iter().enumerate() {
            let hash = hash_fn(item);
            if hash < self.min_hashes[i] {
                self.min_hashes[i] = hash;
            }
        }
    }
    
    fn similarity(&self, other: &MinHasher) -> f64 {
        if self.min_hashes.len() != other.min_hashes.len() {
            return 0.0;
        }
        
        let mut matches = 0;
        for i in 0..self.min_hashes.len() {
            if self.min_hashes[i] == other.min_hashes[i] {
                matches += 1;
            }
        }
        
        matches as f64 / self.min_hashes.len() as f64
    }
}

// Example 38: Compile-Time String Manipulation
// Using const fn for string operations at compile time

const fn const_strlen(s: &str) -> usize {
    s.len()
}

// Define buffer sizes at compile time
const MAX_USERNAME_LEN: usize = const_strlen("administrator") + 10;

struct UserData {
    // Buffer sized exactly right for the maximum username
    username: [u8; MAX_USERNAME_LEN],
    username_len: usize,
}

// Example 39: Time-Memory Tradeoffs for Cryptography
// Using lookup tables to speed up cryptographic operations

fn fast_modular_exponentiation(base: u64, exponent: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    
    // Pre-compute powers of base
    let mut powers = Vec::with_capacity(64);
    let mut current_power = base % modulus;
    powers.push(current_power);
    
    for _ in 1..64 {
        current_power = (current_power * current_power) % modulus;
        powers.push(current_power);
    }
    
    let mut result = 1;
    let mut exp = exponent;
    let mut idx = 0;
    
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * powers[idx]) % modulus;
        }
        exp >>= 1;
        idx += 1;
    }
    
    result
}

// Example 40: Zero-Overhead Type Conversions
// Type conversions without runtime cost

struct Kilometers(f64);
struct Miles(f64);

// Conversion happens at compile time with no runtime overhead
impl From<Kilometers> for Miles {
    fn from(km: Kilometers) -> Self {
        Miles(km.0 * 0.621371)
    }
}

impl From<Miles> for Kilometers {
    fn from(miles: Miles) -> Self {
        Kilometers(miles.0 * 1.60934)
    }
}

fn calculate_distance() {
    let distance_km = Kilometers(100.0);
    // No runtime overhead for this conversion
    let distance_miles: Miles = distance_km.into();
    
    println!("Distance: {} miles", distance_miles.0);
}
